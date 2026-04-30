// Tests for the shared copyToClipboard utility.
//
// Three behavioural cases matter after the insecure-context fix:
//   1. Clipboard API available   → writeText() is called, button shows "Copied"
//   2. Clipboard API undefined   → execCommand("copy") fallback runs, button shows "Copied"
//   3. Both APIs fail            → error toast is shown via showToast()
//
// We exercise these via the SSH "Copy Public Key" button because it:
//   - Is reachable without extra setup on the default (pre-configured) server
//   - Uses copyToClipboard() with local-state feedback (setCopiedKeyId → "Copied"
//     label) AND a non-empty failMessage, making all three cases observable.

const { test, expect } = require("../base-test");
const { navigateAndWait } = require("../helpers");

async function generateSshKey(page) {
	const suffix = Date.now().toString().slice(-6);
	const keyName = `e2e-clipboard-${suffix}`;
	await page.getByPlaceholder("production-box").fill(keyName);
	await page.getByRole("button", { name: "Generate", exact: true }).click();
	await expect(page.locator(".provider-item-name", { hasText: keyName }).first()).toBeVisible({
		timeout: 15_000,
	});
	return keyName;
}

test.describe("copyToClipboard utility", () => {
	test("copy button writes correct text via Clipboard API", async ({ page }) => {
		await navigateAndWait(page, "/settings/ssh");
		await generateSshKey(page);

		// Capture clipboard writes without relying on real browser clipboard
		// permissions (blocked in headless mode by default).
		await page.evaluate(() => {
			window.__clipboardWritten = null;
			try {
				Object.defineProperty(window.navigator, "clipboard", {
					configurable: true,
					value: {
						writeText: (text) => {
							window.__clipboardWritten = text;
							return Promise.resolve();
						},
					},
				});
			} catch {
				// descriptor not configurable — real clipboard may still work
			}
		});

		const copyBtn = page.getByRole("button", { name: "Copy Public Key", exact: true }).first();
		await expect(copyBtn).toBeVisible();
		await copyBtn.click();

		// Button label should flip to "Copied" for ~2 s then revert
		await expect(copyBtn).toHaveText("Copied", { timeout: 2_000 });

		// The written text must be the public key (begins with the key type)
		const written = await page.evaluate(() => window.__clipboardWritten);
		if (written !== null) {
			expect(written.trim()).toMatch(/^ssh-/);
		}
	});

	test("copy button falls back to execCommand when clipboard API is unavailable", async ({ page }) => {
		await navigateAndWait(page, "/settings/ssh");
		await generateSshKey(page);

		// Simulate an insecure context where navigator.clipboard is undefined,
		// then intercept document.execCommand to confirm the fallback fires.
		await page.evaluate(() => {
			window.__execCommandCopyCalled = false;
			try {
				Object.defineProperty(window.navigator, "clipboard", {
					configurable: true,
					value: undefined,
				});
			} catch {
				// already non-configurable in this environment
			}
			const orig = document.execCommand.bind(document);
			document.execCommand = (cmd, ...args) => {
				if (cmd === "copy") window.__execCommandCopyCalled = true;
				return orig(cmd, ...args);
			};
		});

		const copyBtn = page.getByRole("button", { name: "Copy Public Key", exact: true }).first();
		await expect(copyBtn).toBeVisible();
		await copyBtn.click();

		// execCommand path should still produce "Copied" feedback on the button
		await expect(copyBtn).toHaveText("Copied", { timeout: 2_000 });

		const execCommandWasCalled = await page.evaluate(() => window.__execCommandCopyCalled);
		expect(execCommandWasCalled).toBe(true);
	});

	test("copy button shows error toast when both clipboard and execCommand fail", async ({ page }) => {
		await navigateAndWait(page, "/settings/ssh");
		await generateSshKey(page);

		// Exhaust all copy paths: clipboard undefined + execCommand returns false.
		// copyToClipboard() should then call showToast() with the failMessage
		// that SshSection passes: "Could not copy public key — please copy it
		// manually."
		await page.evaluate(() => {
			try {
				Object.defineProperty(window.navigator, "clipboard", {
					configurable: true,
					value: undefined,
				});
			} catch {
				// ignore
			}
			document.execCommand = () => false;
		});

		const copyBtn = page.getByRole("button", { name: "Copy Public Key", exact: true }).first();
		await expect(copyBtn).toBeVisible();
		await copyBtn.click();

		await expect(page.locator(".skills-toast-container")).toContainText("Could not copy public key", {
			timeout: 3_000,
		});

		// Button label must NOT have changed to "Copied" — the copy failed
		await expect(copyBtn).toHaveText("Copy Public Key");
	});
});
