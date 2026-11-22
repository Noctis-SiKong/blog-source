"use strict";
document.documentElement.classList.add("js");

const menuButton = document.getElementById("menu-button");
const mainNav = document.getElementById("site-nav");
const themeButton = document.getElementById("theme-button");
const modes = { auto: "跟随系统", light: "亮色", dark: "暗色" };
const systemDark = window.matchMedia("(prefers-color-scheme: dark)");
function themeLabel() {
  const current = document.documentElement.dataset.theme || "auto";
  const system = systemDark.matches ? "dark" : "light";
  const opposite = system === "light" ? "dark" : "light";
  const next =
    current === "auto" ? opposite : current === opposite ? system : "auto";
  themeButton.dataset.next = next;
  themeButton.title = themeButton.ariaLabel =
    "当前" + modes[current] + "，切换到" + modes[next];
}
themeButton.addEventListener("click", () => {
  const next = themeButton.dataset.next;
  if (next === "auto") delete document.documentElement.dataset.theme;
  else document.documentElement.dataset.theme = next;
  try {
    if (next === "auto") localStorage.removeItem("noctis-theme");
    else localStorage.setItem("noctis-theme", next);
  } catch {}
  themeLabel();
});
systemDark.addEventListener("change", themeLabel);
themeLabel();

function closeMenu() {
  mainNav.classList.remove("is-open");
  menuButton.setAttribute("aria-expanded", "false");
}
menuButton.addEventListener("click", () => {
  menuButton.setAttribute(
    "aria-expanded",
    String(mainNav.classList.toggle("is-open")),
  );
});
document.addEventListener("click", (event) => {
  if (!event.target.closest(".masthead")) closeMenu();
});
document.addEventListener("keydown", (event) => {
  if (event.key === "Escape" && mainNav.classList.contains("is-open")) {
    closeMenu();
    menuButton.focus();
  }
});

// Record only real, same-tab link activations. Direct links and new tabs keep the list fallback.
const entryKey = "noctis-entry";
const state = () => history.state || {};
function noteEntry(values) {
  history.replaceState({ ...state(), ...values }, "");
}
const localPath = (url) => url.pathname + url.search;
const visitType = performance.getEntriesByType("navigation")[0]?.type;
try {
  const source = JSON.parse(sessionStorage.getItem(entryKey));
  sessionStorage.removeItem(entryKey);
  if (
    source &&
    visitType === "navigate" &&
    source.to === localPath(location) &&
    document.referrer &&
    source.from === localPath(new URL(document.referrer)) &&
    history.length > 1
  ) {
    noteEntry({ source: source.from });
  }
} catch {}
function ordinaryClick(event) {
  return (
    event.button === 0 &&
    !event.ctrlKey &&
    !event.metaKey &&
    !event.shiftKey &&
    !event.altKey
  );
}
document.querySelectorAll("[data-return]").forEach((link) => {
  if (!state().source) return;
  link.href = state().source;
  link.querySelector("span").textContent = "返回上一页";
  link.addEventListener("click", (event) => {
    if (!ordinaryClick(event)) return;
    event.preventDefault();
    history.back();
  });
});
document.addEventListener("click", (event) => {
  const link = event.target.closest("a[href]");
  if (
    !link ||
    !ordinaryClick(event) ||
    event.defaultPrevented ||
    link.target ||
    link.hasAttribute("download")
  )
    return;
  const destination = new URL(link.href);
  if (
    destination.origin !== location.origin ||
    localPath(destination) === localPath(location)
  )
    return;
  noteEntry({ scrollY: window.scrollY, focusHref: link.href });
  try {
    sessionStorage.setItem(
      entryKey,
      JSON.stringify({ from: localPath(location), to: localPath(destination) }),
    );
  } catch {}
});
window.addEventListener("pagehide", () =>
  noteEntry({ scrollY: window.scrollY }),
);

async function renderSearch() {
  const page = document.getElementById("search-page");
  if (!page) return;
  const query = (new URLSearchParams(location.search).get("q") || "").trim();
  document.getElementById("query").value = query;
  if (!query) return;
  const status = document.getElementById("search-status");
  status.textContent = "正在搜索…";
  try {
    const response = await fetch(page.dataset.index);
    if (!response.ok) throw new Error("search index unavailable");
    const entries = await response.json();
    const words = query.toLocaleLowerCase().split(/\s+/);
    function rank(entry) {
      const fields = [entry.title, entry.terms.join(" "), entry.text].map(
        (text) => text.toLocaleLowerCase(),
      );
      let score = 0;
      for (const word of words) {
        const match = fields.findIndex((field) => field.includes(word));
        if (match < 0) return 0;
        score += [8, 4, 1][match];
      }
      return score;
    }
    const results = entries
      .map((entry) => ({ entry, score: rank(entry) }))
      .filter((hit) => hit.score)
      .sort((a, b) => b.score - a.score);
    const fragment = document.createDocumentFragment();
    for (const { entry } of results) {
      const row = document.createElement("article");
      row.className = "post-row";
      const date = document.createElement("div");
      date.className = "meta";
      date.textContent = entry.date || "日期未填写";
      const heading = document.createElement("h3");
      const link = document.createElement("a");
      link.href = entry.url;
      link.textContent = entry.title;
      const excerpt = document.createElement("p");
      excerpt.textContent = entry.summary;
      heading.append(link);
      row.append(date, heading, excerpt);
      fragment.append(row);
    }
    document.getElementById("results").replaceChildren(fragment);
    status.textContent = results.length
      ? "找到 " + results.length + " 篇文章"
      : "没有找到匹配的文章，试试其他关键词。";
  } catch {
    status.textContent = "文章索引暂时无法读取，请重新搜索。";
  }
}
const searchReady = renderSearch();
window.addEventListener("pageshow", async (event) => {
  closeMenu();
  try {
    const saved = localStorage.getItem("noctis-theme");
    if (saved === "light" || saved === "dark")
      document.documentElement.dataset.theme = saved;
    else delete document.documentElement.dataset.theme;
    themeLabel();
  } catch {}
  if (!event.persisted && visitType !== "back_forward") return;
  await searchReady;
  const saved = state();
  if (typeof saved.scrollY !== "number") return;
  requestAnimationFrame(() => {
    const previous = [...document.links].find(
      (link) => link.href === saved.focusHref && link.getClientRects().length,
    );
    previous?.focus({ preventScroll: true });
    window.scrollTo({ top: saved.scrollY, behavior: "instant" });
  });
});

const bodyText = document.getElementById("article-body");
if (bodyText) {
  const progress = document.getElementById("progress");
  const percent = document.getElementById("read-percent");
  const tools = document.querySelector(".reader-tools");
  const toc = document.querySelector(".toc");
  const links = [...document.querySelectorAll('.toc-panel a[href^="#"]')];
  const sections = links.map((link) =>
    document.getElementById(decodeURIComponent(link.hash.slice(1))),
  );
  let activeLink;
  let frame;
  let positions = [];
  let start = 0,
    end = 0,
    inset = 0;
  function measure() {
    const box = bodyText.getBoundingClientRect();
    inset =
      document.querySelector(".masthead").offsetHeight +
      tools.offsetHeight +
      16;
    start = window.scrollY + box.top - inset;
    end = window.scrollY + box.bottom - innerHeight;
    positions = sections.map((el) =>
      el ? el.getBoundingClientRect().top + window.scrollY : Infinity,
    );
    schedule();
  }
  function update() {
    frame = 0;
    const value =
      end <= start
        ? window.scrollY >= Math.max(0, end)
          ? 100
          : 0
        : Math.round(
            100 *
              Math.max(
                0,
                Math.min(1, (window.scrollY - start) / (end - start)),
              ),
          );
    progress.value = value;
    percent.textContent = value + "%";
    let current;
    for (let i = 0; i < positions.length; i++) {
      if (positions[i] <= window.scrollY + inset + 3) current = links[i];
    }
    if (current !== activeLink) {
      activeLink?.removeAttribute("aria-current");
      current?.setAttribute("aria-current", "location");
      activeLink = current;
    }
  }
  function schedule() {
    if (!frame) frame = requestAnimationFrame(update);
  }
  window.addEventListener("scroll", schedule, { passive: true });
  window.addEventListener("resize", measure);
  new ResizeObserver(measure).observe(bodyText);
  measure();
  if (toc) {
    const trigger = toc.querySelector("summary");
    toc.addEventListener("click", (event) => {
      const link = event.target.closest("a");
      if (!link || !ordinaryClick(event)) return;
      const heading = document.getElementById(
        decodeURIComponent(link.hash.slice(1)),
      );
      if (!heading) return;
      event.preventDefault();
      toc.open = false;
      // Replace the section fragment so Back still returns to the originating list in one step.
      history.replaceState(state(), "", link.hash);
      heading.tabIndex = -1;
      heading.focus({ preventScroll: true });
      heading.scrollIntoView({ block: "start" });
    });
    document.addEventListener("click", (event) => {
      if (toc.open && !toc.contains(event.target)) toc.open = false;
    });
    document.addEventListener("keydown", (event) => {
      if (event.key === "Escape" && toc.open) {
        toc.open = false;
        trigger.focus();
      }
    });
  }
}
