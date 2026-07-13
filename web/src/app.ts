type Thumb = { t_ms: number; path: string; url: string };
type FactTrack = {
  color: string;
  shape: string;
  direction: string;
  displacement: number;
  start_t_ms: number;
  end_t_ms: number;
};
type Scene = {
  video_id: string;
  scene_id: number;
  start_ms: number;
  end_ms: number;
  keyframe_url: string;
  thumbs: Thumb[];
  caption: string;
  transcript: string;
  facts: { object_counts?: Record<string, number>; tracks?: FactTrack[]; [key: string]: unknown };
};
type VideoRecord = {
  video_id: string;
  title: string;
  video_url: string;
  scene_count: number;
  duration_ms: number;
};
type Citation = {
  video_id: string;
  scene_id: number;
  start_ms: number;
  end_ms: number;
  frames: Thumb[];
};
type RetrievalHit = Scene & {
  score: number;
  ranks: Record<string, number>;
  scores: Record<string, number>;
};
type Answer = {
  question: string;
  answer: string;
  category: string;
  citations: Citation[];
  evidence: Record<string, unknown>;
  retrieval: RetrievalHit[];
  confidence: number;
  provider: string;
  grounding: { start_ms?: number; end_ms?: number; score?: number; method?: string };
};
type Health = {
  ok: boolean;
  index: { videos: number; scenes: number; spaces: string[] };
  synthesizer: string;
  grounder: string;
};

const EXAMPLES = [
  "How many red rectangles are visible in the scene about signal?",
  "Does the yellow rectangle move from left to right in the scene about signal?",
  "What color dominates the background of the scene about ripple?",
  "Is there a yellow rectangle in the scene about signal?",
];

const videoSelect = must<HTMLSelectElement>("video-select");
const player = must<HTMLVideoElement>("player");
const strip = must<HTMLDivElement>("scene-strip");
const form = must<HTMLFormElement>("ask-form");
const questionInput = must<HTMLInputElement>("question");
const askButton = must<HTMLButtonElement>("ask-button");
const statusBox = must<HTMLDivElement>("status");
const answerCard = must<HTMLElement>("answer-card");
const evidence = must<HTMLDivElement>("evidence");
const healthBox = must<HTMLDivElement>("health");
const examplesBox = must<HTMLDivElement>("examples");
const providerHint = must<HTMLSpanElement>("provider-hint");

let videos: VideoRecord[] = [];
let currentVideoId = "";
const sceneCache = new Map<string, Scene[]>();
let activeSceneKey = "";

function must<T extends HTMLElement>(id: string): T {
  const node = document.getElementById(id);
  if (!node) throw new Error(`missing element #${id}`);
  return node as T;
}

function el<K extends keyof HTMLElementTagNameMap>(
  tag: K,
  className?: string,
  text?: string,
): HTMLElementTagNameMap[K] {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function formatMs(ms: number): string {
  const safe = Math.max(0, Math.round(ms));
  const seconds = Math.floor(safe / 1000);
  const millis = safe % 1000;
  const minutes = Math.floor(seconds / 60);
  const hours = Math.floor(minutes / 60);
  const mm = String(minutes % 60).padStart(2, "0");
  const ss = String(seconds % 60).padStart(2, "0");
  const rest = String(millis).padStart(3, "0");
  return hours > 0 ? `${hours}:${mm}:${ss}.${rest}` : `${mm}:${ss}.${rest}`;
}

function setStatus(message: string, isError = false): void {
  statusBox.textContent = message;
  statusBox.className = isError ? "status error" : "status";
}

async function fetchJson<T>(url: string): Promise<T> {
  const response = await fetch(url);
  const payload = (await response.json()) as Record<string, unknown>;
  if (!response.ok) {
    const message = typeof payload.error === "string" ? payload.error : `HTTP ${response.status}`;
    throw new Error(message);
  }
  return payload as T;
}

function keyOf(videoId: string, sceneId: number): string {
  return `${videoId}:${sceneId}`;
}

async function loadHealth(): Promise<Health | null> {
  try {
    const health = await fetchJson<Health>("/api/health");
    healthBox.textContent = `${health.index.scenes} scenes · ${health.index.videos} videos · ${health.synthesizer}/${health.grounder}`;
    healthBox.className = "health ok";
    providerHint.textContent = `${health.synthesizer} synth · ${health.grounder} ground`;
    return health;
  } catch (error) {
    healthBox.textContent = "index unavailable";
    healthBox.className = "health bad";
    return null;
  }
}

async function loadVideos(): Promise<void> {
  const data = await fetchJson<{ videos: VideoRecord[] }>("/api/videos");
  videos = data.videos;
  videoSelect.replaceChildren();
  for (const video of videos) {
    const option = el("option", undefined, video.title || video.video_id);
    option.value = video.video_id;
    videoSelect.append(option);
  }
  if (!videos.length) {
    setStatus("No indexed videos. Run ./verify.sh, then the corpus and ingest commands from the README.", true);
    return;
  }
  await selectVideo(videos[0].video_id);
}

async function loadScenes(videoId: string): Promise<Scene[]> {
  const cached = sceneCache.get(videoId);
  if (cached) return cached;
  const data = await fetchJson<{ scenes: Scene[] }>(`/api/videos/${encodeURIComponent(videoId)}/scenes`);
  sceneCache.set(videoId, data.scenes);
  return data.scenes;
}

async function selectVideo(videoId: string): Promise<void> {
  currentVideoId = videoId;
  const record = videos.find((video) => video.video_id === videoId);
  if (record) {
    player.src = record.video_url;
    player.load();
  }
  videoSelect.value = videoId;
  const scenes = await loadScenes(videoId);
  renderStrip(scenes);
  if (activeSceneKey && !activeSceneKey.startsWith(`${videoId}:`)) {
    activeSceneKey = "";
  }
}

function renderStrip(scenes: Scene[]): void {
  strip.replaceChildren();
  if (!scenes.length) {
    strip.append(el("div", "empty", "No scenes in this video."));
    return;
  }
  for (const scene of scenes) {
    const button = el("button", "scene-card");
    button.type = "button";
    button.dataset.key = keyOf(scene.video_id, scene.scene_id);
    if (button.dataset.key === activeSceneKey) button.classList.add("active");
    const image = el("img");
    image.src = scene.keyframe_url;
    image.alt = `Scene ${scene.scene_id}`;
    image.loading = "lazy";
    const meta = el("div", "scene-meta");
    const title = el("strong", undefined, `Scene ${scene.scene_id}`);
    meta.append(title, el("div", undefined, `${formatMs(scene.start_ms)} – ${formatMs(scene.end_ms)}`));
    button.append(image, meta);
    button.title = scene.caption;
    button.addEventListener("click", () => {
      seekTo(scene.video_id, scene.start_ms);
      setActiveScene(scene.video_id, scene.scene_id);
    });
    strip.append(button);
  }
}

function setActiveScene(videoId: string, sceneId: number): void {
  activeSceneKey = keyOf(videoId, sceneId);
  for (const card of Array.from(strip.querySelectorAll<HTMLButtonElement>(".scene-card"))) {
    card.classList.toggle("active", card.dataset.key === activeSceneKey);
  }
}

async function seekTo(videoId: string, ms: number): Promise<void> {
  if (videoId !== currentVideoId) {
    await selectVideo(videoId);
  }
  const apply = (): void => {
    player.currentTime = Math.max(0, ms / 1000);
    void player.play().catch(() => undefined);
  };
  if (player.readyState >= 1) apply();
  else player.addEventListener("loadedmetadata", apply, { once: true });
}

function renderAnswer(answer: Answer): void {
  answerCard.replaceChildren();
  answerCard.classList.remove("hidden");
  answerCard.append(el("div", "answer-label", "ANSWER"));

  const text = el("div", "answer-text", answer.answer);
  const meta = el("div", "answer-meta");
  const category = el("span", `pill ${answer.category}`, answer.category);
  meta.append(
    category,
    el("span", "pill accent", `confidence ${(answer.confidence * 100).toFixed(0)}%`),
    el("span", "pill", answer.provider),
  );
  if (answer.grounding.start_ms !== undefined && answer.grounding.end_ms !== undefined) {
    meta.append(
      el(
        "span",
        "pill",
        `window ${formatMs(answer.grounding.start_ms)} – ${formatMs(answer.grounding.end_ms)}`,
      ),
    );
  }
  answerCard.append(text, meta);

  const citations = el("div", "citation-list");
  for (const citation of answer.citations) {
    const button = el("button", "citation");
    button.type = "button";
    const left = el("span");
    left.append(
      el("div", "scene", `Scene ${citation.scene_id} · ${citation.video_id}`),
      el(
        "div",
        "time",
        `${formatMs(citation.start_ms)} – ${formatMs(citation.end_ms)}`,
      ),
    );
    button.append(left, el("span", "scene", "jump →"));
    button.addEventListener("click", () => {
      void seekTo(citation.video_id, citation.start_ms);
      setActiveScene(citation.video_id, citation.scene_id);
    });
    citations.append(button);
  }
  answerCard.append(citations);

  const frames = answer.citations.flatMap((citation) => citation.frames).slice(0, 8);
  if (frames.length) {
    const label = el("div", "hint", "Frame previews");
    label.style.marginTop = "11px";
    const frameStrip = el("div", "frame-strip");
    for (const frame of frames) {
      const button = el("button", "frame");
      button.type = "button";
      const image = el("img");
      image.src = frame.url;
      image.alt = `Frame at ${formatMs(frame.t_ms)}`;
      image.loading = "lazy";
      button.append(image, el("span", undefined, formatMs(frame.t_ms)));
      button.addEventListener("click", () => {
        const citation = answer.citations.find((item) => item.frames.some((f) => f.t_ms === frame.t_ms));
        if (citation) {
          void seekTo(citation.video_id, frame.t_ms);
          setActiveScene(citation.video_id, citation.scene_id);
        }
      });
      frameStrip.append(button);
    }
    answerCard.append(label, frameStrip);
  }
}

function renderEvidence(answer: Answer): void {
  evidence.replaceChildren();
  if (!answer.retrieval.length) {
    evidence.append(el("div", "empty", "No retrieved scenes."));
    return;
  }
  for (const hit of answer.retrieval) {
    const card = el("div", "evidence-card");
    const top = el("div", "top");
    top.append(
      el("span", "scene-id", `${hit.video_id} · scene ${hit.scene_id}`),
      el("span", "score", `rrf ${hit.score.toFixed(4)}`),
    );
    const times = el("div", "hint", `${formatMs(hit.start_ms)} – ${formatMs(hit.end_ms)}`);
    const caption = el("p", undefined, hit.caption || "No caption");
    const ranks = el("div", "ranks");
    const labels: Record<string, string> = { caption: "caption", keyframe: "frame", transcript: "asr", lexical: "keyword" };
    for (const [space, rank] of Object.entries(hit.ranks)) {
      ranks.append(el("span", "rank", `${labels[space] ?? space} #${rank}`));
    }
    if (hit.transcript) {
      card.title = hit.transcript;
    }
    card.append(top, times, caption, ranks);
    card.addEventListener("click", () => {
      void seekTo(hit.video_id, hit.start_ms);
      setActiveScene(hit.video_id, hit.scene_id);
    });
    evidence.append(card);
  }
}

async function askQuestion(question: string): Promise<void> {
  if (!question.trim()) return;
  askButton.disabled = true;
  setStatus("Searching scenes, grounding the window, synthesizing…");
  try {
    const answer = await fetchJson<Answer>(`/api/ask?q=${encodeURIComponent(question)}&k=5`);
    renderAnswer(answer);
    renderEvidence(answer);
    const top = answer.retrieval[0];
    if (top) {
      await seekTo(top.video_id, top.start_ms);
      setActiveScene(top.video_id, top.scene_id);
    }
    setStatus(
      `Retrieved ${answer.retrieval.length} scenes · grounded ${formatMs(answer.grounding.start_ms ?? 0)} – ${formatMs(answer.grounding.end_ms ?? 0)}`,
    );
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    setStatus(message, true);
  } finally {
    askButton.disabled = false;
  }
}

function wireExamples(): void {
  for (const example of EXAMPLES) {
    const chip = el("button", "chip", example);
    chip.type = "button";
    chip.addEventListener("click", () => {
      questionInput.value = example;
      void askQuestion(example);
    });
    examplesBox.append(chip);
  }
}

async function init(): Promise<void> {
  wireExamples();
  form.addEventListener("submit", (event) => {
    event.preventDefault();
    void askQuestion(questionInput.value);
  });
  videoSelect.addEventListener("change", () => {
    void selectVideo(videoSelect.value);
  });
  await loadHealth();
  try {
    await loadVideos();
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    setStatus(message, true);
  }
}

void init();
