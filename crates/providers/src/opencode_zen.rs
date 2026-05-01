//! OpenCode Zen provider — opencode.ai's curated multi-protocol model proxy.
//!
//! Zen routes requests to different wire formats based on model family:
//! - `claude-*`  → Anthropic Messages API (`/zen/v1/messages`)
//! - `gpt-*`     → OpenAI Responses API   (`/zen/v1/responses`)
//! - everything  → OpenAI ChatCompletions  (`/zen/v1/chat/completions`)
//!
//! All three paths share a single `OPENCODE_ZEN_API_KEY` and base URL.

use std::{collections::HashMap, pin::Pin, sync::Arc, time::Duration};

use {
    async_trait::async_trait,
    moltis_agents::model::{
        ChatMessage, CompletionResponse, LlmProvider, ModelMetadata, ReasoningEffort, StreamEvent,
    },
    moltis_config::WireApi,
    secrecy::Secret,
    tokio_stream::Stream,
};

use crate::{anthropic::AnthropicProvider, openai::OpenAiProvider};

pub const OPENCODE_ZEN_DEFAULT_BASE_URL: &str = "https://opencode.ai/zen/v1";

/// Static fallback model catalog for when live discovery is unavailable.
/// Model IDs must match what the Zen API accepts — they are the same IDs the
/// underlying provider expects (no `opencode/` prefix).
pub(crate) const OPENCODE_ZEN_MODELS: &[(&str, &str)] = &[
    // OpenAI (Responses API)
    ("gpt-4o", "GPT-4o (OpenCode Zen)"),
    ("gpt-4.1", "GPT-4.1 (OpenCode Zen)"),
    // Anthropic (Messages API)
    ("claude-opus-4-6", "Claude Opus 4.6 (OpenCode Zen)"),
    ("claude-sonnet-4-6", "Claude Sonnet 4.6 (OpenCode Zen)"),
    ("claude-haiku-4-5-20251001", "Claude Haiku 4.5 (OpenCode Zen)"),
    // Gemini (ChatCompletions fallback)
    ("gemini-2.5-pro-preview-05-06", "Gemini 2.5 Pro (OpenCode Zen)"),
    ("gemini-2.5-flash-preview-05-20", "Gemini 2.5 Flash (OpenCode Zen)"),
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

/// An OpenCode Zen (opencode.ai) provider that dispatches to the correct wire format.
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
    ///
    /// `global_cw` and `provider_cw` are the context-window override maps from
    /// `[models.<id>]` and `[providers.opencode-zen.model_overrides]` config respectively.
    pub fn new(
        api_key: Secret<String>,
        model_id: String,
        base_url: String,
        global_cw: HashMap<String, u32>,
        provider_cw: HashMap<String, u32>,
    ) -> Self {
        let inner: Arc<dyn LlmProvider> = match classify_model(&model_id) {
            ZenWireFormat::OpenAiResponses => Arc::new(
                OpenAiProvider::new_with_name(
                    api_key,
                    model_id.clone(),
                    base_url,
                    "opencode-zen".into(),
                )
                .with_wire_api(WireApi::Responses)
                .with_context_window_overrides(global_cw, provider_cw),
            ),
            ZenWireFormat::Anthropic => {
                // AnthropicProvider appends `/v1/messages` to its base_url, so
                // strip the trailing `/v1` from the Zen base URL.
                let anthropic_base = base_url
                    .trim_end_matches('/')
                    .strip_suffix("/v1")
                    .map(str::to_string)
                    .unwrap_or(base_url);
                Arc::new(
                    AnthropicProvider::with_alias(
                        api_key,
                        model_id.clone(),
                        anthropic_base,
                        Some("opencode-zen".into()),
                    )
                    .with_context_window_overrides(global_cw, provider_cw),
                )
            },
            ZenWireFormat::ChatCompletions => Arc::new(
                OpenAiProvider::new_with_name(
                    api_key,
                    model_id.clone(),
                    base_url,
                    "opencode-zen".into(),
                )
                .with_context_window_overrides(global_cw, provider_cw),
            ),
        };
        Self { model_id, inner }
    }
}

#[async_trait]
impl LlmProvider for ZenProvider {
    fn name(&self) -> &str {
        "opencode-zen"
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn opencode_zen_models_not_empty() {
        assert!(!OPENCODE_ZEN_MODELS.is_empty());
    }

    #[test]
    fn opencode_zen_models_have_unique_ids() {
        let mut ids: Vec<&str> = OPENCODE_ZEN_MODELS.iter().map(|(id, _)| *id).collect();
        ids.sort();
        let before = ids.len();
        ids.dedup();
        assert_eq!(before, ids.len(), "duplicate OPENCODE_ZEN_MODELS IDs");
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

    fn dummy_key() -> Secret<String> {
        Secret::new("test-key".into())
    }

    #[test]
    fn zen_provider_name_is_opencode_zen_for_all_wire_formats() {
        let base = "https://opencode.ai/zen/v1".to_string();
        for model_id in &[
            "gpt-4o",
            "claude-sonnet-4-6",
            "gemini-2.5-pro-preview-05-06",
        ] {
            let p = ZenProvider::new(
                dummy_key(),
                model_id.to_string(),
                base.clone(),
                HashMap::new(),
                HashMap::new(),
            );
            assert_eq!(
                p.name(),
                "opencode-zen",
                "name() should be 'opencode-zen' for {model_id}"
            );
        }
    }

    #[test]
    fn context_window_overrides_applied_to_anthropic_inner() {
        let mut provider_cw = HashMap::new();
        provider_cw.insert("claude-sonnet-4-6".to_string(), 50_000u32);
        let p = ZenProvider::new(
            dummy_key(),
            "claude-sonnet-4-6".into(),
            "https://opencode.ai/zen/v1".into(),
            HashMap::new(),
            provider_cw,
        );
        assert_eq!(p.context_window(), 50_000);
    }

    #[test]
    fn context_window_overrides_applied_to_openai_inner() {
        let mut global_cw = HashMap::new();
        global_cw.insert("gpt-4o".to_string(), 64_000u32);
        let p = ZenProvider::new(
            dummy_key(),
            "gpt-4o".into(),
            "https://opencode.ai/zen/v1".into(),
            global_cw,
            HashMap::new(),
        );
        assert_eq!(p.context_window(), 64_000);
    }

    #[test]
    fn context_window_overrides_applied_to_chat_completions_inner() {
        let mut provider_cw = HashMap::new();
        provider_cw.insert("gemini-2.5-flash-preview-05-20".to_string(), 32_000u32);
        let p = ZenProvider::new(
            dummy_key(),
            "gemini-2.5-flash-preview-05-20".into(),
            "https://opencode.ai/zen/v1".into(),
            HashMap::new(),
            provider_cw,
        );
        assert_eq!(p.context_window(), 32_000);
    }

    #[test]
    fn anthropic_routing_with_trailing_slash_base_url() {
        // Verify construction succeeds and name is correct when base URL has trailing slash.
        // The /v1/ suffix must be stripped before AnthropicProvider appends /v1/messages.
        let p = ZenProvider::new(
            dummy_key(),
            "claude-opus-4-6".into(),
            "https://opencode.ai/zen/v1/".into(),
            HashMap::new(),
            HashMap::new(),
        );
        assert_eq!(p.name(), "opencode-zen");
    }

    #[test]
    fn prefixed_model_ids_fall_through_to_chat_completions() {
        // If Zen's /models endpoint returns "anthropic/claude-..." or "openai/gpt-..." style IDs,
        // classify_model will not match the gpt-/claude- prefixes and will route via ChatCompletions.
        // This is the known risk documented in the code review. If that happens in practice, add
        // prefix-stripping normalization here and in fetch_models_from_api.
        assert!(matches!(
            classify_model("anthropic/claude-sonnet-4-6"),
            ZenWireFormat::ChatCompletions
        ));
        assert!(matches!(
            classify_model("openai/gpt-4o"),
            ZenWireFormat::ChatCompletions
        ));
    }
}
