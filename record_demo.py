"""Record a video walkthrough of the SecureFlow demo on GitHub.

Uses direct URL navigation (no fragile CSS selectors) so every scene
is deterministic. Output: a .webm video in ./evidence/.
"""

import asyncio
from playwright.async_api import async_playwright

REPO = "https://github.com/trwilcoxson/secureflow"
VIDEO_DIR = "./demo"


async def scene(page, url, label, scrolls=None, pause=2500):
    """Navigate to a URL, pause, then scroll through the page."""
    print(f"  >> {label}")
    await page.goto(url, wait_until="networkidle")
    await page.wait_for_timeout(pause)
    for dy in (scrolls or []):
        await page.evaluate(f"window.scrollBy(0, {dy})")
        await page.wait_for_timeout(2000)


async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)
        context = await browser.new_context(
            viewport={"width": 1280, "height": 800},
            record_video_dir=VIDEO_DIR,
            record_video_size={"width": 1280, "height": 800},
        )
        page = await context.new_page()
        print("Recording started...")

        # 1. Repo landing -- README with Mermaid diagram
        await scene(page, REPO, "Repo landing + README",
                    scrolls=[400, 600, 400])

        # 2. Issues list -- feature requests + auto-created risk issues
        await scene(page, f"{REPO}/issues", "Issues list",
                    scrolls=[300, 300])

        # 3. Feature request: Payment Processing (#19) -- shows triage comment
        await scene(page, f"{REPO}/issues/19",
                    "Feature request #19 (Payment Processing)",
                    scrolls=[500, 500, 400])

        # 4. Auto-created security risk issue (#26)
        await scene(page, f"{REPO}/issues/26",
                    "Security risk issue #26",
                    scrolls=[400, 400])

        # 5. Auto-created privacy risk issue (#27)
        await scene(page, f"{REPO}/issues/27",
                    "Privacy risk issue #27",
                    scrolls=[400, 400])

        # 6. Auto-created GRC risk issue (#28)
        await scene(page, f"{REPO}/issues/28",
                    "GRC risk issue #28",
                    scrolls=[400, 400])

        # 7. CSS-only change (#21) -- no risk issues created (correct GO)
        await scene(page, f"{REPO}/issues/21",
                    "CSS banner change #21 (no risk)",
                    scrolls=[300])

        # 8. Actions tab -- workflow run list
        await scene(page, f"{REPO}/actions", "Actions tab",
                    scrolls=[200])

        # 9. Click into a workflow run
        run_link = page.locator('a[href*="/actions/runs/"]').first
        if await run_link.count() > 0:
            await run_link.click()
            await page.wait_for_load_state("networkidle")
            await page.wait_for_timeout(3000)
            await page.evaluate("window.scrollBy(0, 300)")
            await page.wait_for_timeout(2000)

        # 10. Back to repo landing
        await scene(page, REPO, "Back to repo landing", pause=2000)

        # Finalize video
        await context.close()
        await browser.close()

    print("Done! Video saved to ./demo/")


if __name__ == "__main__":
    asyncio.run(main())
