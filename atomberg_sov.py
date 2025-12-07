import os
import re
import json
import asyncio
import csv
from collections import defaultdict
from typing import List, Dict, Any, Tuple
from datetime import datetime, timezone

from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from dotenv import load_dotenv

# GitHub Models client from the course stack
from autogen_core.models import UserMessage, SystemMessage
from autogen_ext.models.azure import AzureAIChatCompletionClient
from azure.core.credentials import AzureKeyCredential


# =========================
# 1. Load environment vars
# =========================

load_dotenv()

YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")
if not YOUTUBE_API_KEY:
    raise RuntimeError("YOUTUBE_API_KEY not found in .env")

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
if not GITHUB_TOKEN:
    raise RuntimeError("GITHUB_TOKEN not found in .env")

# =========================
# 2. API clients
# =========================

# YouTube Data API client
youtube = build("youtube", "v3", developerKey=YOUTUBE_API_KEY)

# GitHub Models (gpt-4o-mini) client
gh_client = AzureAIChatCompletionClient(
    model="gpt-4o-mini",
    endpoint="https://models.inference.ai.azure.com",
    credential=AzureKeyCredential(GITHUB_TOKEN),
    model_info={
        "json_output": False,
        "function_calling": False,
        "vision": False,
        "family": "unknown",
    },
)


def load_internal_docs(path: str = "internal_data/product_docs.txt") -> str:
    """Load internal Atomberg product docs from a text file."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        print(f"Warning: internal docs not found at {path}")
        return ""


# =========================
# 3. Config
# =========================

KEYWORDS = [
    "smart fan",
    "bldc fan",
    "smart ceiling fan",
    "energy saving fan",
]

N_YOUTUBE = 30  # top N videos per keyword

BRANDS = {
    "atomberg": ["atomberg", "atom berg"],
    "havells": ["havells"],
    "crompton": ["crompton"],
    "orient": ["orient", "orient electric"],
    "usha": ["usha"],
    "bajaj": ["bajaj"],
    "luminous": ["luminous"],
}

POSITIVE_WORDS = [
    "good",
    "great",
    "love",
    "awesome",
    "amazing",
    "best",
    "excellent",
    "efficient",
    "silent",
    "quiet",
    "worth",
    "recommended",
    "happy",
    "satisfied",
    "energy saving",
    "saves electricity",
]
NEGATIVE_WORDS = [
    "bad",
    "worst",
    "hate",
    "problem",
    "issues",
    "noisy",
    "noise",
    "disappointed",
    "waste",
    "poor",
    "slow",
    "not good",
    "not worth",
]

TOPICS = {
    "energy_saving": ["bldc", "energy", "saving", "electricity", "bill", "watt"],
    "noise_silent": ["silent", "quiet", "noise", "noiseless"],
    "smart_features": ["smart", "wifi", "wi-fi", "alexa", "google home", "app", "remote"],
    "price_value": ["price", "expensive", "cheap", "value for money", "budget"],
}

RETAILER_KEYWORDS = [
    "flipkart",
    "amazon",
    "croma",
    "reliance",
    "vijay sales",
    "tatacliq",
]



# =========================
# 4. Helpers: brands, sentiment, topics, creator type
# =========================

def detect_brands(text: str) -> Dict[str, int]:
    """Count how many times each brand appears in the text."""
    text_l = text.lower()
    counts = {b: 0 for b in BRANDS}
    for brand, aliases in BRANDS.items():
        for alias in aliases:
            matches = re.findall(r"\b" + re.escape(alias) + r"\b", text_l)
            counts[brand] += len(matches)
    return counts


def sentiment_score(text: str) -> float:
    """
    Very simple rule-based sentiment.
    Returns a score between -1 (negative) and +1 (positive).
    """
    text_l = text.lower()
    pos = sum(text_l.count(w) for w in POSITIVE_WORDS)
    neg = sum(text_l.count(w) for w in NEGATIVE_WORDS)
    if pos == 0 and neg == 0:
        return 0.0
    return (pos - neg) / max(pos + neg, 1)


def detect_topics(text: str) -> List[str]:
    text_l = text.lower()
    topics: List[str] = []
    for topic, words in TOPICS.items():
        if any(w in text_l for w in words):
            topics.append(topic)
    return topics


def classify_creator(channel_name: str, title: str) -> str:
    cn = channel_name.lower()
    tl = title.lower()

    if "atomberg" in cn:
        return "brand_atomberg"

    for brand, aliases in BRANDS.items():
        if brand == "atomberg":
            continue
        for alias in aliases:
            if alias in cn:
                return "brand_competitor"

    for kw in RETAILER_KEYWORDS:
        if kw in cn or kw in tl:
            return "retailer"

    return "independent_creator"


def compute_engagement_per_day(rec: Dict[str, Any]) -> float:
    """Normalize engagement by video age (engagement per day)."""
    views = rec["views"]
    likes = rec["likes"]
    comments_count = rec["comments_count"]
    raw = views + 10 * likes + 20 * comments_count

    published_at = rec.get("published_at")
    days = 1
    if published_at:
        try:
            dt = datetime.fromisoformat(published_at.replace("Z", "+00:00"))
            delta = datetime.now(timezone.utc) - dt
            days = max(delta.days, 1)
        except Exception:
            days = 1
    return raw / days


# =========================
# 5. YouTube: comments + search
# =========================

def fetch_comments(video_id: str, max_comments: int = 50) -> List[Dict[str, Any]]:
    comments: List[Dict[str, Any]] = []
    next_page_token = None

    while len(comments) < max_comments:
        try:
            response = youtube.commentThreads().list(
                part="snippet",
                videoId=video_id,
                maxResults=min(50, max_comments - len(comments)),
                pageToken=next_page_token,
                textFormat="plainText",
            ).execute()
        except HttpError as e:
            # Some videos have comments disabled or restricted; skip them.
            print(f"  Skipping comments for video {video_id}: {e}")
            break

        for item in response.get("items", []):
            top = item["snippet"]["topLevelComment"]["snippet"]
            comments.append(
                {
                    "text": top.get("textDisplay", ""),
                    "like_count": int(top.get("likeCount", 0)),
                }
            )

        next_page_token = response.get("nextPageToken")
        if not next_page_token:
            break

    return comments


def search_youtube(keyword: str, max_results: int = N_YOUTUBE) -> List[Dict[str, Any]]:
    # 1. Search for videos
    search_response = youtube.search().list(
        q=keyword,
        part="id",
        type="video",
        maxResults=max_results,
    ).execute()

    video_ids = [item["id"]["videoId"] for item in search_response.get("items", [])]
    if not video_ids:
        return []

    # 2. Get video details
    videos_response = youtube.videos().list(
        id=",".join(video_ids),
        part="snippet,statistics",
    ).execute()

    records: List[Dict[str, Any]] = []

    for item in videos_response.get("items", []):
        vid = item["id"]
        snippet = item["snippet"]
        stats = item.get("statistics", {})

        title = snippet.get("title", "")
        description = snippet.get("description", "")
        channel = snippet.get("channelTitle", "")
        published_at = snippet.get("publishedAt", "")

        views = int(stats.get("viewCount", 0))
        likes = int(stats.get("likeCount", 0))
        comments_count = int(stats.get("commentCount", 0))

        topics = detect_topics(title + "\n" + description)
        creator_type = classify_creator(channel, title)

        # 3. Fetch top-level comments (limited)
        comments = fetch_comments(vid, max_comments=50)

        records.append(
            {
                "platform": "youtube",
                "keyword": keyword,
                "video_id": vid,
                "title": title,
                "description": description,
                "channel": channel,
                "published_at": published_at,
                "creator_type": creator_type,
                "topics": topics,
                "views": views,
                "likes": likes,
                "comments_count": comments_count,
                "comments": comments,
            }
        )

    return records


# =========================
# 6. Compute SoV metrics + extras
# =========================

def compute_sov(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    # Overall brand SoV
    post_mentions = defaultdict(int)
    total_posts_with_any_brand = 0

    engagement_per_brand = defaultdict(float)
    engagement_total = 0.0

    comment_mentions = defaultdict(int)
    positive_voice = defaultdict(float)
    total_positive_voice = 0.0

    # Independent creators vs brand/retailer
    independent_post_mentions = defaultdict(int)
    independent_total_posts = 0

    # Topic-level metrics
    topic_post_mentions: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    topic_total_posts: Dict[str, int] = defaultdict(int)
    topic_positive_voice: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
    topic_positive_total: Dict[str, float] = defaultdict(float)

    # High-risk / opportunity candidates
    high_risk_candidates: List[Dict[str, Any]] = []
    opportunity_candidates: List[Dict[str, Any]] = []

    # Co-mention network
    co_mentions: Dict[Tuple[str, str], int] = defaultdict(int)

    for rec in records:
        post_text = f"{rec['title']}\n{rec['description']}"
        brand_counts_post = detect_brands(post_text)
        topics = rec.get("topics", [])
        creator_type = rec.get("creator_type", "independent_creator")

        any_brand_here = any(c > 0 for c in brand_counts_post.values())
        if any_brand_here:
            total_posts_with_any_brand += 1
            for t in topics:
                topic_total_posts[t] += 1

        engagement_day = compute_engagement_per_day(rec)

        # Overall + indep + topic-level post counts
        for brand, count in brand_counts_post.items():
            if count > 0:
                post_mentions[brand] += 1
                engagement_per_brand[brand] += engagement_day
                engagement_total += engagement_day

                if creator_type == "independent_creator":
                    independent_post_mentions[brand] += 1

                for t in topics:
                    topic_post_mentions[t][brand] += 1

        if any_brand_here and creator_type == "independent_creator":
            independent_total_posts += 1

        # Comment-level analysis
        atomberg_comment_sent_sum = 0.0
        atomberg_comment_count = 0
        brands_mentioned_anywhere = {
            b for b, c in brand_counts_post.items() if c > 0
        }

        for c in rec["comments"]:
            text = c["text"]
            brand_counts_comment = detect_brands(text)
            s = sentiment_score(text)

            present = [b for b, cnt in brand_counts_comment.items() if cnt > 0]

            # Co-mentions: pairs of brands in the same comment
            if len(present) >= 2:
                present_sorted = sorted(present)
                for i in range(len(present_sorted)):
                    for j in range(i + 1, len(present_sorted)):
                        pair = (present_sorted[i], present_sorted[j])
                        co_mentions[pair] += 1

            for brand, count in brand_counts_comment.items():
                if count > 0:
                    comment_mentions[brand] += count
                    brands_mentioned_anywhere.add(brand)

                    # For high-risk detection: track Atomberg sentiment
                    if brand == "atomberg":
                        atomberg_comment_sent_sum += s
                        atomberg_comment_count += 1

                    if s > 0:
                        positive_voice[brand] += s
                        total_positive_voice += s

                        for t in topics:
                            topic_positive_voice[t][brand] += s
                            topic_positive_total[t] += s

        # High-risk: Atomberg mentioned in comments with negative avg sentiment + decent engagement
        if atomberg_comment_count > 0:
            avg_sent = atomberg_comment_sent_sum / atomberg_comment_count
            high_risk_candidates.append(
                {
                    "video_id": rec["video_id"],
                    "title": rec["title"],
                    "url": f"https://www.youtube.com/watch?v={rec['video_id']}",
                    "keyword": rec["keyword"],
                    "engagement_per_day": engagement_day,
                    "avg_atomberg_sentiment": avg_sent,
                    "topics": topics,
                }
            )

        # High-opportunity: competitors mentioned, Atomberg not mentioned anywhere
        if "atomberg" not in brands_mentioned_anywhere:
            competitors = [b for b in brands_mentioned_anywhere if b != "atomberg"]
            if competitors:
                opportunity_candidates.append(
                    {
                        "video_id": rec["video_id"],
                        "title": rec["title"],
                        "url": f"https://www.youtube.com/watch?v={rec['video_id']}",
                        "keyword": rec["keyword"],
                        "engagement_per_day": engagement_day,
                        "competitors": competitors,
                        "topics": topics,
                    }
                )

    # =========================
    # Aggregate metrics
    # =========================

    brands = list(BRANDS.keys())
    result: Dict[str, Any] = {"brands": brands, "metrics": {}, "topic_metrics": {}}

    for brand in brands:
        # Content SoV (all creators)
        if total_posts_with_any_brand > 0:
            sov_content = post_mentions[brand] / total_posts_with_any_brand
        else:
            sov_content = 0.0

        # Engagement SoV (per-day normalized)
        if engagement_total > 0:
            sov_engagement = engagement_per_brand[brand] / engagement_total
        else:
            sov_engagement = 0.0

        # Comment SoV
        total_comment_mentions = sum(comment_mentions.values())
        if total_comment_mentions > 0:
            sov_comments = comment_mentions[brand] / total_comment_mentions
        else:
            sov_comments = 0.0

        # Share of Positive Voice (overall)
        if total_positive_voice > 0:
            sopv = positive_voice[brand] / total_positive_voice
        else:
            sopv = 0.0

        # Content SoV among independent creators only
        if independent_total_posts > 0:
            sov_content_indep = independent_post_mentions[brand] / independent_total_posts
        else:
            sov_content_indep = 0.0

        result["metrics"][brand] = {
            "posts_with_brand": post_mentions[brand],
            "sov_content": sov_content,
            "sov_engagement": sov_engagement,
            "sov_comments": sov_comments,
            "share_of_positive_voice": sopv,
            "posts_with_brand_independent": independent_post_mentions[brand],
            "sov_content_independent": sov_content_indep,
        }

    # Topic-level metrics (SoV + SoPV per topic)
    for topic, brand_counts in topic_post_mentions.items():
        topic_res: Dict[str, Any] = {}
        total_posts_topic = topic_total_posts[topic]
        pos_total_topic = topic_positive_total.get(topic, 0.0)

        for brand in brands:
            posts_b = brand_counts.get(brand, 0)
            if total_posts_topic > 0:
                sov_content_topic = posts_b / total_posts_topic
            else:
                sov_content_topic = 0.0

            if pos_total_topic > 0:
                sopv_topic = topic_positive_voice[topic].get(brand, 0.0) / pos_total_topic
            else:
                sopv_topic = 0.0

            topic_res[brand] = {
                "posts_with_brand": posts_b,
                "sov_content": sov_content_topic,
                "share_of_positive_voice": sopv_topic,
            }

        result["topic_metrics"][topic] = topic_res

    # High-risk & high-opportunity shortlists
    high_risk_sorted = sorted(
        high_risk_candidates,
        key=lambda x: (x["avg_atomberg_sentiment"], -x["engagement_per_day"]),
    )
    high_risk_top = high_risk_sorted[:3]

    opportunity_sorted = sorted(
        opportunity_candidates,
        key=lambda x: -x["engagement_per_day"],
    )
    opportunity_top = opportunity_sorted[:3]

    result["high_risk_videos"] = high_risk_top
    result["high_opportunity_videos"] = opportunity_top

    # Co-mention network (serialize tuples as strings)
    co_mentions_serializable = {
        f"{a}|{b}": count for (a, b), count in co_mentions.items()
    }
    result["co_mentions"] = co_mentions_serializable

    return result


# =========================
# 7. RAG helpers
# =========================

def build_rag_context(
    records: List[Dict[str, Any]],
    internal_docs: str,
    top_k: int = 20,
) -> str:
    """
    Simple retrieval: build a context string from:
    - internal docs
    - top-k most engaged videos (by engagement/day) and a few comments.
    No embeddings, no FAISS – just heuristic retrieval.
    """
    chunks: List[str] = []

    if internal_docs.strip():
        chunks.append("INTERNAL_DOCS:\n" + internal_docs)

    # Sort videos by engagement/day
    sorted_recs = sorted(records, key=compute_engagement_per_day, reverse=True)

    for rec in sorted_recs[:top_k]:
        base = (
            f"VIDEO TITLE: {rec['title']}\n"
            f"DESCRIPTION: {rec['description']}\n"
            f"CHANNEL: {rec['channel']}\n"
            f"KEYWORD: {rec['keyword']}\n"
            f"TOPICS: {', '.join(rec.get('topics', []))}\n"
        )
        chunks.append(base)

        # Add up to 3 comments per video
        for c in rec["comments"][:3]:
            chunks.append(f"COMMENT on '{rec['title']}': {c['text']}")

    return "\n\n---\n\n".join(chunks)



# =========================
# 8. AI insights with GitHub model
# =========================

async def generate_ai_insights(
    sov_results: Dict[str, Any],
    internal_docs: str,
    rag_context: str,
) -> None:
    """
    Use a GitHub model (gpt-4o-mini) to turn numeric SoV metrics + extras into
    clear insights + marketing recommendations for Atomberg.
    """
    metrics_json = json.dumps(sov_results, indent=2)

    system_text = """
You are a senior marketing analyst for Atomberg, a smart/BLDC fan brand in India.
You are given quantitative Share of Voice (SoV) metrics from YouTube search
results for smart fan related keywords.

Your job:
1. Explain what the numbers say about Atomberg vs competitors.
2. Highlight where Atomberg is strong vs weak (content, engagement, comments, sentiment).
3. Comment on topics: energy saving, noise/silence, smart features, price/value.
4. Use the high-risk and high-opportunity video lists to suggest concrete actions
   (e.g., videos to respond to, themes for new content, comparisons to produce).
5. Suggest 4–6 very concrete content & marketing actions Atomberg should take.

Be concise but insightful. Assume the reader is the Atomberg marketing team.
"""

    user_text = f"""
Here are the computed metrics and extra analyses as JSON:

{metrics_json}

Here are internal Atomberg product documents and positioning notes:

{internal_docs}

Here are retrieved snippets (internal + YouTube) that seem relevant
to Atomberg's brand perception and content strategy:

{rag_context}

Key fields in the JSON:
- metrics[brand]: overall SoV and Share of Positive Voice
- topic_metrics[topic][brand]: SoV and Share of Positive Voice per topic
- high_risk_videos: list of videos with high engagement and negative Atomberg sentiment
- high_opportunity_videos: competitor videos with no Atomberg mention but good engagement
- co_mentions: brand pairs that are often mentioned together

Please:
- Summarize Atomberg's overall position vs each key competitor.
- Highlight how Atomberg does in each topic (energy_saving, noise_silent, smart_features, price_value).
- Use internal docs to align recommendations with Atomberg's strengths.
- Use high_risk_videos to recommend specific mitigation actions.
- Use high_opportunity_videos and co_mentions to recommend specific new content ideas.
"""

    messages = [
        SystemMessage(content=system_text.strip(), source="system"),
        UserMessage(content=user_text.strip(), source="user"),
    ]

    print("\n=== AI-Generated Insights & Recommendations ===\n")
    response = await gh_client.create(messages=messages)
    print(response.content)


# =========================
# 9. Exports for the team
# =========================

def export_records_to_csv(records: List[Dict[str, Any]], path: str) -> None:
    fieldnames = [
        "platform",
        "keyword",
        "video_id",
        "title",
        "channel",
        "published_at",
        "creator_type",
        "topics",
        "views",
        "likes",
        "comments_count",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rec in records:
            writer.writerow(
                {
                    "platform": rec["platform"],
                    "keyword": rec["keyword"],
                    "video_id": rec["video_id"],
                    "title": rec["title"],
                    "channel": rec["channel"],
                    "published_at": rec["published_at"],
                    "creator_type": rec.get("creator_type", ""),
                    "topics": ";".join(rec.get("topics", [])),
                    "views": rec["views"],
                    "likes": rec["likes"],
                    "comments_count": rec["comments_count"],
                }
            )


def export_sov_to_json(sov_results: Dict[str, Any], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(sov_results, f, indent=2)


# =========================
# 10. Main
# =========================

def main():
    all_records: List[Dict[str, Any]] = []

    for kw in KEYWORDS:
        print(f"\n=== Collecting for keyword: '{kw}' ===")
        recs = search_youtube(kw, max_results=N_YOUTUBE)
        print(f"  Found {len(recs)} videos.")
        all_records.extend(recs)

    print(f"\nTotal videos collected across keywords: {len(all_records)}")

    sov_results = compute_sov(all_records)

    # Overall SoV table
    print("\n=== Share of Voice Metrics (Overall) ===")
    for brand, m in sov_results["metrics"].items():
        print(f"\nBrand: {brand.upper()}")
        print(f"  Posts with brand (all):            {m['posts_with_brand']}")
        print(f"  SoV (content, all):                {m['sov_content']:.2%}")
        print(f"  SoV (engagement-weighted, all):    {m['sov_engagement']:.2%}")
        print(f"  SoV (comments, all):               {m['sov_comments']:.2%}")
        print(f"  Share of positive voice (all):     {m['share_of_positive_voice']:.2%}")
        print(f"  Posts with brand (independent):    {m['posts_with_brand_independent']}")
        print(f"  SoV (content, independent only):   {m['sov_content_independent']:.2%}")

    # Shortlists
    print("\nTop high-risk videos (negative Atomberg sentiment, high engagement):")
    for v in sov_results["high_risk_videos"]:
        print(
            f"- {v['title']} | {v['url']} | kw={v['keyword']} | "
            f"eng/day={v['engagement_per_day']:.1f} | "
            f"avg Atomberg sentiment={v['avg_atomberg_sentiment']:.2f} | "
            f"topics={','.join(v['topics'])}"
        )

    print("\nTop high-opportunity videos (competitor focus, no Atomberg, high engagement):")
    for v in sov_results["high_opportunity_videos"]:
        print(
            f"- {v['title']} | {v['url']} | kw={v['keyword']} | "
            f"eng/day={v['engagement_per_day']:.1f} | "
            f"competitors={','.join(v['competitors'])} | "
            f"topics={','.join(v['topics'])}"
        )

    # Exports
    export_records_to_csv(all_records, "atomberg_youtube_data.csv")
    export_sov_to_json(sov_results, "atomberg_sov_summary.json")
    print("\nExported:")
    print("  - atomberg_youtube_data.csv")
    print("  - atomberg_sov_summary.json")

    # Build simple RAG-style context (no embeddings, no FAISS)
    internal_docs = load_internal_docs()
    rag_context = build_rag_context(all_records, internal_docs, top_k=20)

    # AI narrative + recommendations
    asyncio.run(generate_ai_insights(sov_results, internal_docs, rag_context))



if __name__ == "__main__":
    main()
