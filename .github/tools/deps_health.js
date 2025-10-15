/* eslint-disable no-console */
/**
 * Dependency Health Analyzer (Node script)
 * ---------------------------------------
 * Reads deps.json + audit.json (created by earlier steps),
 * fetches PyPI + GitHub metadata, computes a health score,
 * and posts a Markdown report:
 *   - PR event  -> comment on the PR
 *   - non-PR    -> create or update a rolling Issue labeled "dependency-health"
 *
 * Requires Node 20+ (built-in fetch).
 */

/**
 * @typedef {Object} Vulnerability
 * @property {string=} id          // advisory id (e.g., GHSA-..., PYSEC-..., CVE-...)
 * @property {string=} advisory    // summary
 * @property {string=} severity    // LOW / MEDIUM / HIGH / CRITICAL
 * @property {string=} fix_version // first fixed version if known
 * @property {string=} cve         // CVE id if provided
 */

/**
 * @typedef {Object} Health
 * @property {string} name
 * @property {number} score
 * @property {Record<string, any>} metrics
 * @property {string[]} warnings
 * @property {string[]} suggestions
 * @property {Vulnerability[]=} vulns
 */

/// -------------------- Imports & FS helpers --------------------
const fs = require("fs");
const path = require("path");

function readJSONSafe(filePath) {
  try {
    const raw = fs.readFileSync(filePath, "utf8");
    return JSON.parse(raw);
  } catch {
    return null;
  }
}

// Small utilities used by the Main section
function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

function chunk(arr, size) {
  const out = [];
  for (let i = 0; i < arr.length; i += size) out.push(arr.slice(i, i + size));
  return out;
}

// Split oversized comments to stay under GitHub limits (~65k)
async function postLargeComment(issueNumber, body) {
  const LIMIT = 60000; // safety margin
  if (body.length <= LIMIT) {
    await createIssueComment(issueNumber, body);
    return;
  }
  const parts = [];
  for (let i = 0; i < body.length; i += LIMIT) parts.push(body.slice(i, i + LIMIT));
  for (let idx = 0; idx < parts.length; idx++) {
    const header = parts.length > 1 ? `**Part ${idx + 1}/${parts.length}**\n\n` : '';
    await createIssueComment(issueNumber, header + parts[idx]);
  }
}

/// -------------------- Env + Context --------------------
/** owner/repo */
const repoSlug = process.env.GITHUB_REPOSITORY || "";
const [OWNER, REPO] = repoSlug.split("/");
/** GitHub token for REST calls */
const GITHUB_TOKEN = process.env.GITHUB_TOKEN || process.env.GITHUB_PAT || process.env.INPUT_GITHUB_TOKEN || "";
/** event name & payload */
const EVENT_NAME = process.env.GITHUB_EVENT_NAME || "";
const EVENT_PATH = process.env.GITHUB_EVENT_PATH || "";
const EVENT = readJSONSafe(EVENT_PATH) || {};

/** hard fail early with clear messages */
if (!OWNER || !REPO) {
  console.error("❌ GITHUB_REPOSITORY is missing or malformed (expected owner/repo).");
  process.exit(1);
}
if (!GITHUB_TOKEN) {
  console.error("❌ GITHUB_TOKEN not provided. Pass it via env in the workflow step.");
  process.exit(1);
}

/// -------------------- Input files --------------------
const depsPath = path.join(process.cwd(), "deps.json");
const auditPath = path.join(process.cwd(), "audit.json");

const depsJson = readJSONSafe(depsPath) || { direct: [] };
const auditJson = readJSONSafe(auditPath) || [];

/** index audit by package name (lowercase) */
const auditIndex = new Map();
for (const item of auditJson) {
  const k = (item?.name || "").toLowerCase();
  if (!k) continue;
  if (!auditIndex.has(k)) auditIndex.set(k, []);
  auditIndex.get(k).push({
    id: item.advisory?.id,
    advisory: item.advisory?.summary,
    severity: item.advisory?.severity,
    fix_version: item.fix_versions?.[0],
    cve: item.advisory?.cve,
  });
}

const MAX = Number(process.env.DEP_HEALTH_MAX || Number.POSITIVE_INFINITY);
const allDeps = Array.from(new Set(depsJson.direct || [])); // de-dup just in case
const DEPENDENCIES = Number.isFinite(MAX) ? allDeps.slice(0, MAX) : allDeps;
console.log(`🔎 Analyzing ${DEPENDENCIES.length} dependencies (MAX=${MAX})`);

console.log(`🔎 Analyzing ${DEPENDENCIES.length} dependencies for ${OWNER}/${REPO} (event: ${EVENT_NAME})`);

/// -------------------- HTTP helpers (GitHub & PyPI) --------------------
/**
 * @param {string} url
 * @param {RequestInit=} init
 */
async function ghFetch(url, init) {
  const res = await fetch(url, {
    ...init,
    headers: {
      ...(init?.headers || {}),
      Authorization: `Bearer ${GITHUB_TOKEN}`,
      Accept: "application/vnd.github+json",
      "X-GitHub-Api-Version": "2022-11-28",
      "User-Agent": "deps-health-check",
    },
  });
  return res;
}

/** GET JSON or throw with status for debugging */
async function ghGetJSON(url) {
  const res = await ghFetch(url);
  if (!res.ok) {
    const txt = await res.text();
    throw new Error(`GitHub GET ${url} -> ${res.status} ${res.statusText} :: ${txt}`);
  }
  return res.json();
}

/** POST JSON body; returns parsed JSON (or {} if none) */
async function ghPostJSON(url, bodyObj) {
  const res = await ghFetch(url, {
    method: "POST",
    body: JSON.stringify(bodyObj),
    headers: { "Content-Type": "application/json" },
  });
  if (!res.ok) {
    const txt = await res.text();
    throw new Error(`GitHub POST ${url} -> ${res.status} ${res.statusText} :: ${txt}`);
  }
  try {
    return await res.json();
  } catch {
    return {};
  }
}

/** List open issues filtered by label */
async function listOpenIssuesByLabel(label) {
  const url = `https://api.github.com/repos/${OWNER}/${REPO}/issues?state=open&labels=${encodeURIComponent(label)}`;
  return ghGetJSON(url);
}

/** Create issue */
async function createIssue(title, body, labels = []) {
  const url = `https://api.github.com/repos/${OWNER}/${REPO}/issues`;
  return ghPostJSON(url, { title, body, labels });
}

/** Comment on issue/PR (issues and PRs share the same comments endpoint) */
async function createIssueComment(issueNumber, body) {
  const url = `https://api.github.com/repos/${OWNER}/${REPO}/issues/${issueNumber}/comments`;
  return ghPostJSON(url, { body });
}

/// -------------------- Analysis --------------------
/**
 * @param {string} pkg
 * @returns {Promise<Health>}
 */
async function analyzeDependency(pkg) {
  /** @type {Health} */
  const health = { name: pkg, score: 0, metrics: {}, warnings: [], suggestions: [], vulns: [] };

  // Merge pip-audit data
  const v = auditIndex.get(pkg) || [];
  if (v.length) {
    health.vulns = v;
    health.metrics.vulnerabilityCount = v.length;
  }

  // PyPI metadata (recency, releases, license)
  try {
    const pypi = await fetch(`https://pypi.org/pypi/${encodeURIComponent(pkg)}/json`);
    if (!pypi.ok) {
      health.warnings.push("Package not found on PyPI");
      return health;
    }
    const pypiData = await pypi.json();
    const info = pypiData.info || {};
    const releases = Object.keys(pypiData.releases || {});
    const latest = info.version;
    const uploaded = pypiData.releases?.[latest]?.[0]?.upload_time;
    const lastDate = uploaded ? new Date(uploaded) : null;
    const daysSinceRel = lastDate ? Math.floor((Date.now() - lastDate.getTime()) / 86400000) : null;

    health.metrics.latestVersion = latest;
    health.metrics.totalReleases = releases.length;
    if (daysSinceRel !== null) health.metrics.daysSinceLastRelease = daysSinceRel;

    // Recency score (30 pts)
    let recency = 30;
    if (daysSinceRel === null) {
      recency = 10;
      health.warnings.push("⚠️ No release timestamp found");
    } else if (daysSinceRel > 730) {
      recency = 5;
      health.warnings.push("⚠️ No release in over 2 years");
    } else if (daysSinceRel > 365) {
      recency = 15;
      health.warnings.push("⚠️ No release in over 1 year");
    } else if (daysSinceRel > 180) {
      recency = 25;
    }
    health.score += recency;

    if (!info.license) health.warnings.push("⚠️ No license specified");
    if (releases.length < 5) health.warnings.push("⚠️ Few releases (early stage?)");

    // Try to infer GitHub repo from project URLs
    const candidates = [
      info.home_page,
      info.project_url,
      info.bugtrack_url,
      ...(info.project_urls ? Object.values(info.project_urls) : []),
    ].filter(Boolean);

    let gh = null;
    for (const u of candidates) {
      const m = String(u).match(/github\.com\/([^\/]+\/[^\/#]+)/i);
      if (m) {
        gh = m[1].replace(/\.git$/, "");
        break;
      }
    }

    // GitHub repo metrics (stars/activity/issues)
    if (gh) {
      const [owner, repo] = gh.split("/");
      try {
        const data = await ghGetJSON(`https://api.github.com/repos/${owner}/${repo}`);
        const stars = data.stargazers_count ?? 0;
        const forks = data.forks_count ?? 0;
        const openIssues = data.open_issues_count ?? 0;
        const pushedAt = data.pushed_at ? new Date(data.pushed_at) : null;
        const daysSincePush = pushedAt ? Math.floor((Date.now() - pushedAt.getTime()) / 86400000) : null;

        Object.assign(health.metrics, {
          githubUrl: data.html_url,
          stars,
          forks,
          openIssues,
          daysSinceLastCommit: daysSincePush,
        });

        // Stars (20 pts)
        const starsScore = Math.min(20, (stars / 1000) * 20);
        health.score += starsScore;

        // Activity (30 pts)
        let act = 30;
        if (daysSincePush === null) {
          act = 10;
        } else if (daysSincePush > 365) {
          act = 5;
          health.warnings.push("⚠️ No commits in over 1 year");
        } else if (daysSincePush > 180) {
          act = 15;
        } else if (daysSincePush > 90) {
          act = 25;
        }
        health.score += act;

        // Issue ratio (20 pts)
        const ratio = openIssues / Math.max(stars, 1);
        let issueScore = 20;
        if (ratio > 0.5) {
          issueScore = 5;
          health.warnings.push("⚠️ High ratio of open issues");
        } else if (ratio > 0.2) {
          issueScore = 15;
        }
        health.score += issueScore;
      } catch (e) {
        console.info(`GitHub fetch failed for ${gh}: ${e.message}`);
      }
    }
  } catch (e) {
    health.warnings.push(`PyPI fetch error: ${e.message}`);
  }

  // Suggestions (security + overall)
  if ((health.metrics.vulnerabilityCount ?? 0) > 0) {
    health.suggestions.push("Review and patch known vulnerabilities (see below).");
  }
  if (health.score < 40) {
    health.suggestions.push("Consider alternatives; project may be inactive or risky.");
  } else if (health.score < 60) {
    health.suggestions.push("Monitor for updates and consider pinning/constraints.");
  }

  return health;
}

/// -------------------- Report rendering --------------------
/**
 * @param {Health} dep
 * @param {boolean} compact
 */
function formatDependency(dep, compact = false) {
  let out = `### ${dep.name}\n\n`;
  out += `**Health Score:** ${Math.round(dep.score)}/100 `;
  out += dep.score >= 70 ? "✅\n\n" : dep.score >= 40 ? "⚠️\n\n" : "🚨\n\n";

  if (!compact && dep.metrics.githubUrl) out += `**Repository:** ${dep.metrics.githubUrl}\n\n`;

  out += "**Metrics:**\n";
  if (dep.metrics.latestVersion) out += `- Version: ${dep.metrics.latestVersion}\n`;
  if (dep.metrics.daysSinceLastRelease !== undefined)
    out += `- Last release: ${dep.metrics.daysSinceLastRelease} days ago\n`;
  if (dep.metrics.daysSinceLastCommit !== undefined)
    out += `- Last commit: ${dep.metrics.daysSinceLastCommit} days ago\n`;
  if (dep.metrics.stars !== undefined) out += `- GitHub stars: ${dep.metrics.stars}\n`;
  if (dep.metrics.openIssues !== undefined) out += `- Open issues: ${dep.metrics.openIssues}\n`;
  if (dep.metrics.vulnerabilityCount !== undefined)
    out += `- Vulnerabilities: ${dep.metrics.vulnerabilityCount}\n`;

  // Security details
  if (!compact && dep.vulns && dep.vulns.length) {
    out += "\n**Vulnerabilities:**\n";
    for (const v of dep.vulns) {
      const id = v.id || v.cve || "Advisory";
      const sev = v.severity ? ` (${v.severity})` : "";
      const fix = v.fix_version ? ` → fix: ${v.fix_version}` : "";
      out += `- ${id}${sev}: ${v.advisory || ""}${fix}\n`;
    }
  }

  if (dep.warnings.length && !compact) {
    out += "\n**Warnings:**\n";
    for (const w of dep.warnings) out += `- ${w}\n`;
  }

  if (dep.suggestions.length && !compact) {
    out += "\n**Suggestions:**\n";
    for (const s of dep.suggestions) out += `- ${s}\n`;
  }

  out += "\n";
  return out;
}

/**
 * @param {Health[]} results
 */
function buildReport(results) {
  const critical = results.filter((r) => r.score < 40);
  const warning = results.filter((r) => r.score >= 40 && r.score < 70);
  const healthy = results.filter((r) => r.score >= 70);

  let md = "# 🏥 Python Dependency Health Report\n\n";
  md += `**Analysis Date:** ${new Date().toISOString().split("T")[0]}\n\n`;
  md += "## 📊 Summary\n\n";
  md += `- ✅ **Healthy:** ${healthy.length} dependencies\n`;
  md += `- ⚠️ **Warning:** ${warning.length} dependencies\n`;
  md += `- 🚨 **Critical:** ${critical.length} dependencies\n\n`;

  if (critical.length) {
    md += "## 🚨 Critical Dependencies (Score < 40)\n\n";
    for (const d of critical) md += formatDependency(d);
  }
  if (warning.length) {
    md += "## ⚠️ Dependencies Needing Attention (Score 40–69)\n\n";
    for (const d of warning) md += formatDependency(d);
  }
  if (healthy.length) {
    md += "## ✅ Healthy Dependencies (Score 70+)\n\n";
    md += "<details>\n<summary>Click to expand healthy dependencies</summary>\n\n";
    for (const d of healthy) md += formatDependency(d, true);
    md += "</details>\n\n";
  }

  md += "\n---\n*Generated by Dependency Health Checker*";
  return md;
}

/// -------------------- Main --------------------
(async () => {
  try {
    // Concurrency knobs (override via env)
    const BATCH = Number(process.env.DEP_HEALTH_BATCH || 10);     // how many deps at once
    const PAUSE = Number(process.env.DEP_HEALTH_DELAY_MS || 500); // ms between batches

    /** @type {Health[]} */
    const results = [];

    // Process dependencies in parallel batches (polite & fast)
    for (const group of chunk(DEPENDENCIES, BATCH)) {
      const settled = await Promise.allSettled(group.map(d => analyzeDependency(d)));

      for (let i = 0; i < settled.length; i++) {
        const s = settled[i];
        const name = group[i];
        if (s.status === 'fulfilled') {
          results.push(s.value);
        } else {
          console.warn(`Analyze failed for ${name}: ${s.reason?.message || s.reason}`);
        }
      }

      // brief pause between batches to be gentle with APIs
      if (PAUSE > 0) await sleep(PAUSE);
    }

    const report = buildReport(results);

    if (EVENT_NAME === "pull_request") {
      const prNumber = EVENT?.pull_request?.number ?? EVENT?.number; // fallback
      if (!prNumber) throw new Error("PR number not found in event payload.");
      await postLargeComment(prNumber, report); // handles long reports
      console.log(`📝 Commented health report on PR #${prNumber}`);
    } else {
      // scheduled / workflow_dispatch: create or update rolling issue
      const label = "dependency-health";
      const open = await listOpenIssuesByLabel(label);
      if (Array.isArray(open) && open.length > 0) {
        const issueNumber = open[0].number;
        await postLargeComment(issueNumber, `## Updated Health Report\n\n${report}`);
        console.log(`🔁 Updated health report on Issue #${issueNumber}`);
      } else {
        const created = await createIssue("🏥 Dependency Health Report", report, [label]);
        console.log(`🆕 Created Issue #${created.number} with health report`);
      }
    }
  } catch (err) {
    console.error(`❌ Dependency Health Analyzer failed: ${err.message}`);
    process.exit(1);
  }
})();
