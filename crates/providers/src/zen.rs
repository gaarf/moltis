//! Zen provider — opencode.ai's curated multi-protocol model proxy.
//!
//! Zen routes requests to different wire formats based on model family:
//! - `claude-*`  → Anthropic Messages API (`/zen/v1/messages`)
//! - `gpt-*`     → OpenAI Responses API   (`/zen/v1/responses`)
//! - everything  → OpenAI ChatCompletions  (`/zen/v1/chat/completions`)
//!
//! All three paths share a single `ZEN_API_KEY` and base URL.

use std::{pin::Pin, sync::Arc, time::Duration};

use {
    async_trait::async_trait,
    moltis_agents::model::{
        ChatMessage, CompletionResponse, LlmProvider, ModelMetadata, ReasoningEffort, StreamEvent,
    },
    moltis_config::WireApi,
    secrecy::Secret,
    tokio_stream::Stream,
};

use crate::{
    anthropic::AnthropicProvider,
    discovered_model::DiscoveredModel,
    openai::{self, OpenAiProvider},
};

pub const ZEN_DEFAULT_BASE_URL: &str = "https://opencode.ai/zen/v1";

/// Static fallback model catalog for when live discovery is unavailable.
/// Model IDs must match what the Zen API accepts — they are the same IDs the
/// underlying provider expects (no `opencode/` prefix).
pub(crate) const ZEN_MODELS: &[(&str, &str)] = &[
    // OpenAI (Responses API)
    ("gpt-4o", "GPT-4o (Zen)"),
    ("gpt-4.1", "GPT-4.1 (Zen)"),
    // Anthropic (Messages API)
    ("claude-opus-4-6", "Claude Opus 4.6 (Zen)"),
    ("claude-sonnet-4-6", "Claude Sonnet 4.6 (Zen)"),
    ("claude-haiku-4-5-20251001", "Claude Haiku 4.5 (Zen)"),
    // Gemini (ChatCompletions fallback)
    ("gemini-2.5-pro-preview-05-06", "Gemini 2.5 Pro (Zen)"),
    ("gemini-2.5-flash-preview-05-20", "Gemini 2.5 Flash (Zen)"),
];

/// Wire format to use for a given model, determined by model ID prefix.
enum ZenWireFormat {
    /// OpenAI Responses API — `gpt-*` models.
    OpenAiResponses,
    /// Anthropic Messages API — `claude-*` models.
    Anthropic,
    /// OpenAI ChatCompletions — everything else (Gemini, etc.).
    ChatCompletions,
}

fn classify_model(model_id: &str) -> ZenWireFormat {
    if model_id.starts_with("gpt-") {
        ZenWireFormat::OpenAiResponses
    } else if model_id.starts_with("claude-") {
        ZenWireFormat::Anthropic
    } else {
        ZenWireFormat::ChatCompletions
    }
}

/// A Zen (opencode.ai) provider that dispatches to the correct wire format.
///
/// The inner provider is selected once at construction time based on the model
/// ID prefix, so all hot-path method calls are simple `Arc<dyn LlmProvider>`
/// delegate calls with no branching.
pub struct ZenProvider {
    model_id: String,
    inner: Arc<dyn LlmProvider>,
}

impl ZenProvider {
    /// Construct a `ZenProvider` for `model_id`.
    ///
    /// `base_url` should be the `v1` prefix, e.g. `https://opencode.ai/zen/v1`.
    /// Anthropic routing strips the trailing `/v1` because `AnthropicProvider`
    /// appends `/v1/messages` internally.
    pub fn new(api_key: Secret<String>, model_id: String, base_url: String) -> Self {
        let inner: Arc<dyn LlmProvider> = match classify_model(&model_id) {
            ZenWireFormat::OpenAiResponses => Arc::new(
                OpenAiProvider::new_with_name(api_key, model_id.clone(), base_url, "zen".into())
                    .with_wire_api(WireApi::Responses),
            ),
            ZenWireFormat::Anthropic => {
                // AnthropicProvider appends `/v1/messages` to its base_url, so
                // strip the trailing `/v1` from the Zen base URL.
                let anthropic_base = base_url
                    .trim_end_matches('/')
                    .strip_suffix("/v1")
                    .map(str::to_string)
                    .unwrap_or(base_url);
                Arc::new(AnthropicProvider::with_alias(
                    api_key,
                    model_id.clone(),
                    anthropic_base,
                    Some("zen".into()),
                ))
            },
            ZenWireFormat::ChatCompletions => Arc::new(OpenAiProvider::new_with_name(
                api_key,
                model_id.clone(),
                base_url,
                "zen".into(),
            )),
        };
        Self { model_id, inner }
    }
}

#[async_trait]
impl LlmProvider for ZenProvider {
    fn name(&self) -> &str {
        "zen"
    }

    fn id(&self) -> &str {
        &self.model_id
    }

    async fn complete(
        &self,
        messages: &[ChatMessage],
        tools: &[serde_json::Value],
    ) -> anyhow::Result<CompletionResponse> {
        self.inner.complete(messages, tools).await
    }

    fn supports_tools(&self) -> bool {
        self.inner.supports_tools()
    }

    fn context_window(&self) -> u32 {
        self.inner.context_window()
    }

    fn supports_vision(&self) -> bool {
        self.inner.supports_vision()
    }

    fn tool_mode(&self) -> Option<moltis_config::ToolMode> {
        self.inner.tool_mode()
    }

    fn stream(
        &self,
        messages: Vec<ChatMessage>,
    ) -> Pin<Box<dyn Stream<Item = StreamEvent> + Send + '_>> {
        self.inner.stream(messages)
    }

    fn stream_with_tools(
        &self,
        messages: Vec<ChatMessage>,
        tools: Vec<serde_json::Value>,
    ) -> Pin<Box<dyn Stream<Item = StreamEvent> + Send + '_>> {
        self.inner.stream_with_tools(messages, tools)
    }

    fn reasoning_effort(&self) -> Option<ReasoningEffort> {
        self.inner.reasoning_effort()
    }

    fn with_reasoning_effort(
        self: Arc<Self>,
        effort: ReasoningEffort,
    ) -> Option<Arc<dyn LlmProvider>> {
        let new_inner = Arc::clone(&self.inner).with_reasoning_effort(effort)?;
        Some(Arc::new(ZenProvider {
            model_id: self.model_id.clone(),
            inner: new_inner,
        }))
    }

    fn probe_timeout(&self) -> Duration {
        self.inner.probe_timeout()
    }

    async fn check_availability(&self) -> anyhow::Result<()> {
        self.inner.check_availability().await
    }

    async fn model_metadata(&self) -> anyhow::Result<ModelMetadata> {
        self.inner.model_metadata().await
    }
}

/// Fetch live models from `GET /zen/v1/models` (OpenAI-compatible endpoint).
pub async fn fetch_models_from_api(
    api_key: Secret<String>,
    base_url: String,
) -> anyhow::Result<Vec<DiscoveredModel>> {
    openai::fetch_models_from_api(api_key, base_url).await
}

/// Spawn a background thread to fetch Zen models and return a receiver.
pub fn start_model_discovery(
    api_key: Secret<String>,
    base_url: String,
) -> std::sync::mpsc::Receiver<anyhow::Result<Vec<DiscoveredModel>>> {
    openai::start_model_discovery(api_key, base_url)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zen_models_not_empty() {
        assert!(!ZEN_MODELS.is_empty());
    }

    #[test]
    fn zen_models_have_unique_ids() {
        let mut ids: Vec<&str> = ZEN_MODELS.iter().map(|(id, _)| *id).collect();
        ids.sort();
        let before = ids.len();
        ids.dedup();
        assert_eq!(before, ids.len(), "duplicate ZEN_MODELS IDs");
    }

    #[test]
    fn classify_gpt_uses_responses() {
        assert!(matches!(
            classify_model("gpt-4o"),
            ZenWireFormat::OpenAiResponses
        ));
        assert!(matches!(
            classify_model("gpt-4.1"),
            ZenWireFormat::OpenAiResponses
        ));
    }

    #[test]
    fn classify_claude_uses_anthropic() {
        assert!(matches!(
            classify_model("claude-sonnet-4-6"),
            ZenWireFormat::Anthropic
        ));
        assert!(matches!(
            classify_model("claude-opus-4-6"),
            ZenWireFormat::Anthropic
        ));
    }

    #[test]
    fn classify_other_uses_chat_completions() {
        assert!(matches!(
            classify_model("gemini-2.5-pro-preview-05-06"),
            ZenWireFormat::ChatCompletions
        ));
        assert!(matches!(
            classify_model("qwen3-max"),
            ZenWireFormat::ChatCompletions
        ));
    }

    #[test]
    fn anthropic_base_url_strip() {
        // Simulate what new() does: strip /v1 for AnthropicProvider
        let base = "https://opencode.ai/zen/v1";
        let stripped = base
            .trim_end_matches('/')
            .strip_suffix("/v1")
            .map(str::to_string)
            .unwrap_or_else(|| base.to_string());
        assert_eq!(stripped, "https://opencode.ai/zen");
    }

    #[test]
    fn anthropic_base_url_strip_trailing_slash() {
        let base = "https://opencode.ai/zen/v1/";
        let stripped = base
            .trim_end_matches('/')
            .strip_suffix("/v1")
            .map(str::to_string)
            .unwrap_or_else(|| base.to_string());
        assert_eq!(stripped, "https://opencode.ai/zen");
    }
}
