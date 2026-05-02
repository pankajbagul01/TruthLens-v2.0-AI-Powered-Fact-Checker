import streamlit as st
import pandas as pd
import numpy as np
import re
import heapq
import hashlib
import json
import requests
from collections import defaultdict
from datetime import datetime
from difflib import SequenceMatcher
import random
import time

try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False

try:
    from dotenv import load_dotenv
    import os
    load_dotenv()
except ImportError:
    import os

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

# ============================================
# CONFIGURATION
# ============================================
CATEGORIES = ["technology", "science", "health", "business", "environment"]

# ============================================
# UNIT 1: ADVANCED TREES
# ============================================

class Node:
    __slots__ = ('key', 'value', 'left', 'right', 'height', 'color', 'parent', 'thread', 'next')
    def __init__(self, key, value=None):
        self.key = key; self.value = value; self.left = None; self.right = None
        self.height = 1; self.color = 1; self.parent = None; self.thread = False; self.next = None

class AVLTree:
    def __init__(self): self.root = None
    def get_height(self, node): return node.height if node else 0
    def get_balance(self, node): return self.get_height(node.left) - self.get_height(node.right) if node else 0
    def right_rotate(self, y):
        x = y.left; T2 = x.right; x.right = y; y.left = T2
        y.height = 1 + max(self.get_height(y.left), self.get_height(y.right))
        x.height = 1 + max(self.get_height(x.left), self.get_height(x.right)); return x
    def left_rotate(self, x):
        y = x.right; T2 = y.left; y.left = x; x.right = T2
        x.height = 1 + max(self.get_height(x.left), self.get_height(x.right))
        y.height = 1 + max(self.get_height(y.left), self.get_height(y.right)); return y
    def insert(self, key, value=None): self.root = self._insert(self.root, key, value)
    def _insert(self, node, key, value):
        if not node: return Node(key, value)
        if key < node.key: node.left = self._insert(node.left, key, value)
        elif key > node.key: node.right = self._insert(node.right, key, value)
        else: node.value = value; return node
        node.height = 1 + max(self.get_height(node.left), self.get_height(node.right))
        balance = self.get_balance(node)
        if balance > 1 and key < node.left.key: return self.right_rotate(node)
        if balance < -1 and key > node.right.key: return self.left_rotate(node)
        if balance > 1 and key > node.left.key: node.left = self.left_rotate(node.left); return self.right_rotate(node)
        if balance < -1 and key < node.right.key: node.right = self.right_rotate(node.right); return self.left_rotate(node)
        return node
    def search(self, key): return self._search(self.root, key)
    def _search(self, node, key):
        if not node or node.key == key: return node.value if node else None
        return self._search(node.left, key) if key < node.key else self._search(node.right, key)
    def inorder(self):
        result = []; self._inorder(self.root, result); return result
    def _inorder(self, node, result):
        if node: self._inorder(node.left, result); result.append((node.key, node.value)); self._inorder(node.right, result)

class RedBlackTree:
    def __init__(self):
        self.root = None; self.NIL = Node(0); self.NIL.color = 0; self.NIL.left = None; self.NIL.right = None
    def insert(self, key, value=None):
        node = Node(key, value); node.left = self.NIL; node.right = self.NIL
        y = None; x = self.root
        while x != self.NIL and x:
            y = x
            if node.key < x.key: x = x.left
            else: x = x.right
        node.parent = y
        if not y or y == self.NIL: self.root = node
        elif node.key < y.key: y.left = node
        else: y.right = node
        node.color = 1; self._fix_insert(node)
    def _fix_insert(self, k):
        while k.parent and k.parent.color == 1:
            if k.parent == k.parent.parent.left:
                u = k.parent.parent.right
                if u and u.color == 1:
                    k.parent.color = 0; u.color = 0; k.parent.parent.color = 1; k = k.parent.parent
                else:
                    if k == k.parent.right: k = k.parent; self._left_rotate(k)
                    k.parent.color = 0; k.parent.parent.color = 1; self._right_rotate(k.parent.parent)
            else:
                u = k.parent.parent.left
                if u and u.color == 1:
                    k.parent.color = 0; u.color = 0; k.parent.parent.color = 1; k = k.parent.parent
                else:
                    if k == k.parent.left: k = k.parent; self._right_rotate(k)
                    k.parent.color = 0; k.parent.parent.color = 1; self._left_rotate(k.parent.parent)
            if k == self.root: break
        self.root.color = 0
    def _left_rotate(self, x):
        y = x.right; x.right = y.left
        if y.left != self.NIL: y.left.parent = x
        y.parent = x.parent
        if x.parent == self.NIL or not x.parent: self.root = y
        elif x == x.parent.left: x.parent.left = y
        else: x.parent.right = y
        y.left = x; x.parent = y
    def _right_rotate(self, x):
        y = x.left; x.left = y.right
        if y.right != self.NIL: y.right.parent = x
        y.parent = x.parent
        if x.parent == self.NIL or not x.parent: self.root = y
        elif x == x.parent.right: x.parent.right = y
        else: x.parent.left = y
        y.right = x; x.parent = y
    def search(self, key): return self._search(self.root, key)
    def _search(self, node, key):
        if not node or node == self.NIL: return None
        if key == node.key: return node.value
        return self._search(node.left, key) if key < node.key else self._search(node.right, key)

class BTreeNode:
    def __init__(self, t, leaf=False):
        self.t = t; self.leaf = leaf; self.keys = []; self.values = []; self.children = []

class BTree:
    def __init__(self, t=3): self.t = t; self.root = BTreeNode(t, True)
    def insert(self, key, value):
        root = self.root
        if len(root.keys) == (2 * self.t - 1):
            new_root = BTreeNode(self.t, False); new_root.children.insert(0, root)
            self._split_child(new_root, 0); self.root = new_root; self._insert_non_full(new_root, key, value)
        else: self._insert_non_full(root, key, value)
    def _split_child(self, parent, i):
        t = self.t; y = parent.children[i]; z = BTreeNode(t, y.leaf)
        parent.keys.insert(i, y.keys[t-1]); parent.values.insert(i, y.values[t-1]); parent.children.insert(i+1, z)
        z.keys = y.keys[t:]; z.values = y.values[t:]; y.keys = y.keys[:t-1]; y.values = y.values[:t-1]
        if not y.leaf: z.children = y.children[t:]; y.children = y.children[:t]
    def _insert_non_full(self, node, key, value):
        i = len(node.keys) - 1
        if node.leaf:
            node.keys.append(None); node.values.append(None)
            while i >= 0 and key < node.keys[i]:
                node.keys[i+1] = node.keys[i]; node.values[i+1] = node.values[i]; i -= 1
            node.keys[i+1] = key; node.values[i+1] = value
        else:
            while i >= 0 and key < node.keys[i]: i -= 1
            i += 1
            if len(node.children[i].keys) == (2*self.t-1):
                self._split_child(node, i)
                if key > node.keys[i]: i += 1
            self._insert_non_full(node.children[i], key, value)
    def search(self, key): return self._search(self.root, key)
    def _search(self, node, key):
        i = 0
        while i < len(node.keys) and key > node.keys[i]: i += 1
        if i < len(node.keys) and key == node.keys[i]: return node.values[i]
        if node.leaf: return None
        return self._search(node.children[i], key)

class BPlusNode:
    def __init__(self, leaf=True):
        self.leaf = leaf; self.keys = []
        self.values = [] if leaf else None
        self.children = [] if not leaf else None
        self.next = None

class BPlusTree:
    def __init__(self, order=4): self.order = order; self.root = BPlusNode(leaf=True)
    def insert(self, key, value):
        root = self.root
        if len(root.keys) == self.order - 1:
            new_root = BPlusNode(leaf=False); new_root.children.append(self.root)
            self._split_child(new_root, 0); self.root = new_root; self._insert_non_full(self.root, key, value)
        else: self._insert_non_full(root, key, value)
    def _split_child(self, parent, index):
        node = parent.children[index]; mid = len(node.keys) // 2; new_node = BPlusNode(leaf=node.leaf)
        if node.leaf:
            new_node.keys = node.keys[mid:]; new_node.values = node.values[mid:]
            node.keys = node.keys[:mid]; node.values = node.values[:mid]
            new_node.next = node.next; node.next = new_node
            parent.keys.insert(index, new_node.keys[0]); parent.children.insert(index+1, new_node)
        else:
            mid_key = node.keys[mid]; new_node.keys = node.keys[mid+1:]; new_node.children = node.children[mid+1:]
            node.keys = node.keys[:mid]; node.children = node.children[:mid+1]
            parent.keys.insert(index, mid_key); parent.children.insert(index+1, new_node)
    def _insert_non_full(self, node, key, value):
        if node.leaf:
            i = 0
            while i < len(node.keys) and key > node.keys[i]: i += 1
            node.keys.insert(i, key); node.values.insert(i, value)
        else:
            i = 0
            while i < len(node.keys) and key >= node.keys[i]: i += 1
            if len(node.children[i].keys) == self.order - 1:
                self._split_child(node, i)
                if key > node.keys[i]: i += 1
            self._insert_non_full(node.children[i], key, value)
    def search(self, key):
        node = self.root
        while not node.leaf:
            i = 0
            while i < len(node.keys) and key >= node.keys[i]: i += 1
            node = node.children[i]
        for i, k in enumerate(node.keys):
            if k == key: return node.values[i] if node.values else None
        return None

class HeapTree:
    def __init__(self): self.heap = []
    def insert(self, value, data=None): heapq.heappush(self.heap, (value, str(data)))
    def extract_min(self): return heapq.heappop(self.heap) if self.heap else None
    def size(self): return len(self.heap)

class HuffmanNode:
    def __init__(self, char, freq):
        self.char = char; self.freq = freq; self.left = None; self.right = None
    def __lt__(self, other): return self.freq < other.freq

class HuffmanTree:
    def __init__(self): self.codes = {}; self.root = None
    def build(self, text):
        freq = defaultdict(int)
        for c in text: freq[c] += 1
        heap = [HuffmanNode(char, f) for char, f in freq.items()]; heapq.heapify(heap)
        while len(heap) > 1:
            left = heapq.heappop(heap); right = heapq.heappop(heap)
            merged = HuffmanNode(None, left.freq + right.freq); merged.left = left; merged.right = right
            heapq.heappush(heap, merged)
        self.root = heap[0] if heap else None; self._generate_codes(self.root, ""); return self.codes
    def _generate_codes(self, node, code):
        if node:
            if node.char is not None: self.codes[node.char] = code
            else: self._generate_codes(node.left, code+"0"); self._generate_codes(node.right, code+"1")
    def compress(self, text):
        if not self.codes: self.build(text)
        return ''.join(self.codes.get(c, '') for c in text)

class TrieNode:
    def __init__(self): self.children = {}; self.is_end = False; self.value = None

class Trie:
    def __init__(self): self.root = TrieNode()
    def insert(self, word, value=None):
        node = self.root
        for char in word:
            if char not in node.children: node.children[char] = TrieNode()
            node = node.children[char]
        node.is_end = True; node.value = value
    def starts_with(self, prefix):
        node = self._traverse(prefix)
        if not node: return []
        return self._collect_words(node, prefix)
    def _traverse(self, word):
        node = self.root
        for char in word:
            if char not in node.children: return None
            node = node.children[char]
        return node
    def _collect_words(self, node, prefix):
        results = []
        if node.is_end: results.append((prefix, node.value))
        for char, child in node.children.items(): results.extend(self._collect_words(child, prefix+char))
        return results

class BKNode:
    def __init__(self, word, data=None): self.word = word; self.data = data; self.children = {}

class BKTree:
    def __init__(self): self.root = None
    def _levenshtein(self, s1, s2):
        if len(s1) < len(s2): return self._levenshtein(s2, s1)
        if len(s2) == 0: return len(s1)
        prev = range(len(s2)+1)
        for i, c1 in enumerate(s1):
            curr = [i+1]
            for j, c2 in enumerate(s2):
                curr.append(min(prev[j+1]+1, curr[j]+1, prev[j]+(c1!=c2)))
            prev = curr
        return prev[-1]
    def insert(self, word, data=None):
        if not self.root: self.root = BKNode(word, data); return
        node = self.root
        while True:
            dist = self._levenshtein(word, node.word)
            if dist == 0: node.data = data; return
            if dist not in node.children: node.children[dist] = BKNode(word, data); return
            node = node.children[dist]
    def search(self, word, max_dist=2):
        results = []
        if not self.root: return results
        stack = [self.root]
        while stack:
            node = stack.pop(); dist = self._levenshtein(word, node.word)
            if dist <= max_dist: results.append((node.word, node.data, dist))
            for d in range(max(0, dist-max_dist), dist+max_dist+1):
                if d in node.children: stack.append(node.children[d])
        return results

class SkipListNode:
    def __init__(self, key, value=None, level=0):
        self.key = key; self.value = value; self.forward = [None]*(level+1)

class SkipList:
    def __init__(self, max_level=16, p=0.5):
        self.max_level = max_level; self.p = p
        self.header = SkipListNode(-float('inf'), None, max_level); self.level = 0; self.size = 0
    def _random_level(self):
        level = 0
        while random.random() < self.p and level < self.max_level: level += 1
        return level
    def insert(self, key, value=None):
        update = [None]*(self.max_level+1); current = self.header
        for i in range(self.level, -1, -1):
            while current.forward[i] and current.forward[i].key < key: current = current.forward[i]
            update[i] = current
        current = current.forward[0]
        if current and current.key == key: current.value = value
        else:
            new_level = self._random_level()
            if new_level > self.level:
                for i in range(self.level+1, new_level+1): update[i] = self.header
                self.level = new_level
            new_node = SkipListNode(key, value, new_level)
            for i in range(new_level+1): new_node.forward[i] = update[i].forward[i]; update[i].forward[i] = new_node
            self.size += 1

class UnionFind:
    def __init__(self): self.parent = {}; self.rank = {}
    def find(self, x):
        if x not in self.parent: self.parent[x] = x; self.rank[x] = 0; return x
        if self.parent[x] != x: self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    def union(self, x, y):
        px, py = self.find(x), self.find(y)
        if px == py: return False
        if self.rank[px] < self.rank[py]: px, py = py, px
        self.parent[py] = px
        if self.rank[px] == self.rank[py]: self.rank[px] += 1
        return True

class PersistentArray:
    def __init__(self): self.versions = [{}]; self.version_count = 1
    def update(self, index, value, version=-1):
        if version == -1: version = self.version_count - 1
        new_version = self.versions[version].copy(); new_version[index] = value
        self.versions.append(new_version); self.version_count += 1; return self.version_count - 1


# ============================================
# GROQ AI INTEGRATION
# ============================================

def call_groq(prompt: str, api_key: str) -> str:
    if not GROQ_AVAILABLE:
        raise ImportError("groq package not installed. Run: pip install groq")
    client = Groq(api_key=api_key)
    models_to_try = [
        "llama-3.3-70b-versatile",
        "llama-3.1-70b-versatile",
        "llama3-70b-8192",
        "mixtral-8x7b-32768",
        "gemma2-9b-it"
    ]
    last_error = None
    for model_name in models_to_try:
        try:
            completion = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": "You are a precise fact-checking AI. Return ONLY valid JSON, no markdown, no explanations, no backticks."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=2000
            )
            return completion.choices[0].message.content
        except Exception as e:
            last_error = e
            err_str = str(e).lower()
            if "decommissioned" in err_str or "not supported" in err_str or "deprecated" in err_str:
                continue
            raise Exception(f"Groq API error: {str(e)}")
    raise Exception(f"All Groq models failed. Last error: {last_error}")


def safe_parse_json(text: str) -> dict:
    text = text.strip()
    text = re.sub(r'```(?:json)?', '', text).strip()
    try:
        start = text.index('{')
        end = text.rindex('}') + 1
        text = text[start:end]
    except ValueError:
        raise ValueError(f"No JSON object found in response: {text[:200]}")
    return json.loads(text)


def expand_query_with_groq(claim: str, api_key: str) -> dict:
    prompt = f"""Analyze this claim and return ONLY a valid JSON object with these exact fields:

{{
  "primary_query": "best 3-5 word English search query for news",
  "secondary_queries": ["alternative query 1", "alternative query 2"],
  "key_topics": ["topic1", "topic2", "topic3"],
  "claim_type": "health|politics|science|technology|crime|economy|environment|social|other",
  "is_sensational": true or false,
  "normalized_claim": "rewritten claim in clear neutral English"
}}

Claim: "{claim}"

Rules:
- primary_query must be short (3-5 words), factual, suitable for a news search engine
- Do NOT include the year unless it is critical
- Return ONLY the JSON object, no markdown, no backticks, no explanation"""

    text = call_groq(prompt, api_key)
    return safe_parse_json(text)


def analyze_articles_with_groq(claim: str, articles: list, api_key: str) -> dict:
    if not articles:
        raise Exception("No articles were fetched. Check your NewsAPI key and try again.")

    article_context = []
    for i, a in enumerate(articles[:15]):
        title = (a.get('title') or '')[:150]
        desc = (a.get('description') or '')[:200]
        source = a.get('source') or 'Unknown'
        article_context.append(f"[{i}] SOURCE: {source} | TITLE: {title} | DESC: {desc}")

    article_list = "\n".join(article_context)

    prompt = f"""You are an expert fact-checker. Analyze this claim against these real news articles.

CLAIM: "{claim}"

ARTICLES:
{article_list}

Return ONLY a valid JSON object with these exact fields:
{{
  "article_scores": [
    {{
      "index": 0,
      "similarity_score": 0.85,
      "relationship": "confirms|contradicts|partially_confirms|unrelated|context_only",
      "relevance_explanation": "one sentence explanation"
    }}
  ],
  "verdict": "AUTHENTIC|LIKELY_AUTHENTIC|UNVERIFIED|LIKELY_MISINFO|MISINFO|INSUFFICIENT_DATA",
  "confidence": 0.85,
  "verdict_reasoning": "2-3 sentence explanation of why this verdict was reached",
  "red_flags": ["specific red flag 1", "specific red flag 2"],
  "supporting_evidence": ["specific evidence 1", "specific evidence 2"],
  "fact_check_summary": "One clear paragraph summary of findings",
  "recommended_action": "specific action the user should take"
}}

Scoring rules:
- similarity_score: 0.0 to 1.0
- Only include articles with similarity_score > 0.2 in article_scores
- verdict must be based on the actual articles
- Return ONLY the JSON object, no markdown, no backticks, no explanation"""

    text = call_groq(prompt, api_key)
    return safe_parse_json(text)


# ============================================
# NEWS FETCHER
# ============================================

def search_news_multi_query(queries: list, api_key: str, page_size: int = 15) -> list:
    if not api_key:
        raise Exception("NewsAPI key is missing. Please enter your key in the sidebar.")

    all_articles = {}

    for query in queries[:3]:
        if not query or not query.strip():
            continue

        encoded_query = requests.utils.quote(query.strip())
        url = (
            f"https://newsapi.org/v2/everything"
            f"?q={encoded_query}"
            f"&apiKey={api_key}"
            f"&pageSize={page_size}"
            f"&language=en"
            f"&sortBy=relevancy"
        )

        try:
            response = requests.get(url, timeout=15)
        except requests.exceptions.ConnectionError:
            raise Exception("Network error: Could not connect to NewsAPI. Check your internet connection.")
        except requests.exceptions.Timeout:
            raise Exception("NewsAPI request timed out. Try again.")

        if response.status_code == 401:
            raise Exception("❌ Invalid NewsAPI key (401 Unauthorized). Get your free key at newsapi.org/register")
        if response.status_code == 426:
            raise Exception("🔒 NewsAPI plan upgrade required (426). The free developer plan only allows localhost.")
        if response.status_code == 429:
            raise Exception("⏳ NewsAPI rate limit reached (429). You have exceeded your daily request limit.")
        if response.status_code != 200:
            raise Exception(f"NewsAPI error {response.status_code}: {response.text[:300]}")

        data = response.json()

        if data.get('status') == 'error':
            raise Exception(f"NewsAPI returned error: {data.get('message', 'Unknown error')}")

        for article in data.get('articles', []):
            title = article.get('title', '')
            if not title or title == '[Removed]':
                continue
            title_hash = hashlib.md5(title.encode()).hexdigest()
            if title_hash not in all_articles:
                all_articles[title_hash] = {
                    'title': title,
                    'description': article.get('description', ''),
                    'source': article.get('source', {}).get('name', 'Unknown'),
                    'url': article.get('url', ''),
                    'published_at': article.get('publishedAt', datetime.now().isoformat()),
                    'category': 'search_result'
                }

        time.sleep(0.4)

    return list(all_articles.values())


def clean_text(text):
    if not text: return ""
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


# ============================================
# MAIN APPLICATION CLASS
# ============================================

class MisinformationTracker:
    def __init__(self):
        self.avl = AVLTree()
        self.rbt = RedBlackTree()
        self.btree = BTree(t=3)
        self.bplus = BPlusTree(order=4)
        self.heap = HeapTree()
        self.huffman = HuffmanTree()
        self.trie = Trie()
        self.bk = BKTree()
        self.skip = SkipList(max_level=8)
        self.uf = UnionFind()
        self.persistent = PersistentArray()
        self.articles = []
        self.last_update = None

    def add_article(self, title, description, source, category, url="", published_at=""):
        article_id = len(self.articles)
        clean_title = clean_text(title)
        combined_text = clean_text(f"{title} {description}")

        self.articles.append({
            'id': article_id,
            'title': title,
            'clean_title': combined_text,
            'description': description,
            'source': source,
            'category': category,
            'url': url,
            'published_at': published_at
        })

        self.avl.insert(article_id, clean_title)
        self.rbt.insert(article_id, clean_title)
        self.btree.insert(article_id, clean_title)
        self.bplus.insert(article_id, clean_title)
        self.trie.insert(clean_title[:50], article_id)
        self.bk.insert(clean_title[:60], article_id)
        self.skip.insert(article_id, clean_title)
        self.uf.find(article_id)
        return article_id

    def get_statistics(self):
        try:
            return {
                'Total Articles': len(self.articles),
                'AVL Tree Nodes': len(self.avl.inorder()),
                'Skip List Size': self.skip.size,
                'Trie Entries': len(self.trie.starts_with('')),
                'BK-Tree Size': len(self.bk.root.children) if self.bk.root else 0,
                'Last Update': self.last_update.strftime("%H:%M:%S") if self.last_update else "Never"
            }
        except:
            return {'Total Articles': len(self.articles), 'Last Update': "Never"}


# ============================================
# PLOTLY THEME HELPERS
# ============================================

PLOTLY_LAYOUT = dict(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(13,13,24,0.6)',
    font=dict(family='Fragment Mono, monospace', color='#9090b8', size=11),
    title_font=dict(family='Space Grotesk, sans-serif', color='#d4d4f0', size=14),
    margin=dict(l=40, r=20, t=50, b=40),
)

# Default legend style — merge manually to avoid duplicate keyword errors
_DEFAULT_LEGEND = dict(bgcolor='rgba(13,13,24,0.8)', bordercolor='#1e1e35', borderwidth=1)

VERDICT_COLORS = {
    'AUTHENTIC': '#39d98a',
    'LIKELY_AUTHENTIC': '#5dde1a',
    'UNVERIFIED': '#5b8fff',
    'LIKELY_MISINFO': '#f5a623',
    'MISINFO': '#ff4d6d',
    'INSUFFICIENT_DATA': '#5a5a82',
}

CLAIM_TYPE_COLORS = [
    '#5b8fff', '#39d98a', '#ff4d6d', '#f5a623', '#c084fc',
    '#f472b6', '#34d399', '#fb923c', '#60a5fa', '#a78bfa'
]


def render_visualization_tab(history: list):
    """Render the Visualization tab with Plotly charts based on session history."""

    st.markdown("""
    <div style="padding: 8px 0 24px 0;">
        <div style="font-family:'Fragment Mono',monospace; font-size:0.6rem; color:#5a5a82; letter-spacing:0.2em; text-transform:uppercase; margin-bottom:6px;">Session Analytics</div>
        <h2 style="font-family:'Space Grotesk',sans-serif; font-size:1.7rem; font-weight:700; color:#d4d4f0; margin:0; letter-spacing:-0.03em;">
            Intelligence Dashboard
        </h2>
        <p style="color:#5a5a82; font-size:0.75rem; margin-top:6px; font-family:'Fragment Mono',monospace;">
            Real-time graphs from your session's claim analysis history
        </p>
    </div>
    """, unsafe_allow_html=True)

    if not history:
        st.markdown("""
        <div style="background:var(--surface); border:1px solid var(--border2); border-radius:4px; padding:48px; text-align:center; margin-top:20px; position:relative; overflow:hidden;">
            <div style="position:absolute; inset:0; background: radial-gradient(ellipse at center, rgba(91,143,255,0.03) 0%, transparent 70%); pointer-events:none;"></div>
            <div style="font-family:'Bebas Neue',sans-serif; font-size:3rem; color:rgba(91,143,255,0.1); letter-spacing:0.1em; margin-bottom:12px;">NO DATA</div>
            <div style="font-family:'Space Grotesk',sans-serif; font-weight:600; font-size:0.9rem; color:#3a3a58; margin-bottom:8px;">Run a claim analysis first</div>
            <div style="font-size:0.75rem; color:#2a2a48; font-family:'Fragment Mono',monospace; line-height:2;">
                Charts will auto-update after every check.
            </div>
        </div>
        """, unsafe_allow_html=True)
        return

    if not PLOTLY_AVAILABLE:
        st.error("Plotly not installed. Run: `pip install plotly` then restart.")
        return

    # ---- ROW 1: Verdict Distribution + Confidence Over Time ----
    col1, col2 = st.columns(2)

    with col1:
        verdict_counts = defaultdict(int)
        for h in history:
            verdict_counts[h.get('verdict', 'INSUFFICIENT_DATA')] += 1

        labels = list(verdict_counts.keys())
        values = list(verdict_counts.values())
        colors = [VERDICT_COLORS.get(v, '#5a5a82') for v in labels]

        fig_pie = go.Figure(go.Pie(
            labels=labels,
            values=values,
            hole=0.55,
            marker=dict(colors=colors, line=dict(color='#050508', width=2)),
            textfont=dict(family='Fragment Mono, monospace', size=10, color='#d4d4f0'),
            hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Share: %{percent}<extra></extra>',
        ))
        fig_pie.update_layout(
            **PLOTLY_LAYOUT,
            title='Verdict Distribution',
            showlegend=True,
            legend=dict(orientation='v', x=1.02, y=0.5, font=dict(size=9, color='#5a5a82'), bgcolor='rgba(0,0,0,0)', borderwidth=0),
            annotations=[dict(
                text=f'<b>{len(history)}</b><br><span style="font-size:10px">checks</span>',
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=14, color='#d4d4f0', family='Space Grotesk, sans-serif')
            )]
        )
        st.plotly_chart(fig_pie, use_container_width=True, config={'displayModeBar': False})

    with col2:
        confidences = [h.get('confidence', 0) * 100 for h in history]
        indices = list(range(1, len(history) + 1))
        verdicts = [h.get('verdict', 'UNVERIFIED') for h in history]
        point_colors = [VERDICT_COLORS.get(v, '#5a5a82') for v in verdicts]

        fig_conf = go.Figure()
        fig_conf.add_trace(go.Scatter(
            x=indices, y=confidences,
            mode='lines',
            line=dict(color='rgba(0,229,255,0.25)', width=2),
            showlegend=False,
            hoverinfo='skip'
        ))
        fig_conf.add_trace(go.Scatter(
            x=indices, y=confidences,
            mode='markers+text',
            marker=dict(color=point_colors, size=10, line=dict(color='#050508', width=1.5)),
            text=[f"{c:.0f}%" for c in confidences],
            textposition='top center',
            textfont=dict(size=9, color='#5a5a82', family='Fragment Mono, monospace'),
            hovertemplate='<b>Check #%{x}</b><br>Confidence: %{y:.1f}%<extra></extra>',
            showlegend=False
        ))
        fig_conf.update_layout(
            **PLOTLY_LAYOUT,
            title='Confidence Score per Check',
            xaxis_title='Check #',
            yaxis_title='Confidence (%)',
            yaxis=dict(range=[0, 105], gridcolor='#1e1e35', tickfont=dict(color='#5a5a82')),
            xaxis=dict(tickmode='linear', dtick=1, gridcolor='#1e1e35', tickfont=dict(color='#5a5a82')),
        )
        st.plotly_chart(fig_conf, use_container_width=True, config={'displayModeBar': False})

    # ---- ROW 2: Claim Type Bar + Sensational Gauge ----
    col3, col4 = st.columns([2, 1])

    with col3:
        type_counts = defaultdict(int)
        for h in history:
            ct = h.get('claim_type', 'other')
            type_counts[ct] += 1

        sorted_types = sorted(type_counts.items(), key=lambda x: x[1], reverse=True)
        type_labels = [t[0].upper() for t in sorted_types]
        type_values = [t[1] for t in sorted_types]
        bar_colors = CLAIM_TYPE_COLORS[:len(type_labels)]

        fig_bar = go.Figure(go.Bar(
            x=type_values,
            y=type_labels,
            orientation='h',
            marker=dict(
                color=bar_colors,
                line=dict(color='rgba(0,0,0,0)', width=0)
            ),
            text=type_values,
            textposition='outside',
            textfont=dict(size=10, color='#5a5a82', family='Fragment Mono, monospace'),
            hovertemplate='<b>%{y}</b><br>Claims: %{x}<extra></extra>',
        ))
        fig_bar.update_layout(
            **PLOTLY_LAYOUT,
            title='Claims by Type',
            xaxis_title='Count',
            yaxis_title='',
            yaxis=dict(gridcolor='rgba(0,0,0,0)', tickfont=dict(color='#9090b8', size=10)),
            xaxis=dict(gridcolor='#1e1e35', tickfont=dict(color='#5a5a82')),
            bargap=0.3,
        )
        st.plotly_chart(fig_bar, use_container_width=True, config={'displayModeBar': False})

    with col4:
        sensational_count = sum(1 for h in history if h.get('is_sensational', False))
        sensational_pct = (sensational_count / len(history) * 100) if history else 0

        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=sensational_pct,
            number=dict(suffix="%", font=dict(family='Space Grotesk, sans-serif', size=28, color='#d4d4f0')),
            gauge=dict(
                axis=dict(range=[0, 100], tickfont=dict(color='#5a5a82', size=9)),
                bar=dict(color='#f5a623', thickness=0.6),
                bgcolor='#111120',
                borderwidth=1,
                bordercolor='#1e1e35',
                steps=[
                    dict(range=[0, 33], color='rgba(160,255,111,0.12)'),
                    dict(range=[33, 66], color='rgba(255,181,71,0.12)'),
                    dict(range=[66, 100], color='rgba(255,59,107,0.12)'),
                ],
                threshold=dict(line=dict(color='#ff4d6d', width=2), thickness=0.8, value=66)
            ),
            title=dict(text="Sensational Claims", font=dict(family='Space Grotesk, sans-serif', color='#5a5a82', size=12))
        ))
        fig_gauge.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(family='Fragment Mono, monospace', color='#d4d4f0'),
            margin=dict(l=20, r=20, t=60, b=20),
            height=250,
        )
        st.plotly_chart(fig_gauge, use_container_width=True, config={'displayModeBar': False})

    # ---- ROW 3: Articles Per Check + Source Diversity ----
    col5, col6 = st.columns(2)

    with col5:
        articles_per_check = [h.get('articles_fetched', 0) for h in history]
        check_nums = list(range(1, len(history) + 1))

        fig_area = go.Figure()
        fig_area.add_trace(go.Scatter(
            x=check_nums,
            y=articles_per_check,
            fill='tozeroy',
            fillcolor='rgba(0,229,255,0.08)',
            line=dict(color='#5b8fff', width=2),
            mode='lines+markers',
            marker=dict(color='#5b8fff', size=6, line=dict(color='#050508', width=1)),
            hovertemplate='<b>Check #%{x}</b><br>Articles: %{y}<extra></extra>',
            showlegend=False
        ))
        fig_area.update_layout(
            **PLOTLY_LAYOUT,
            title='Articles Fetched per Check',
            xaxis_title='Check #',
            yaxis_title='Article Count',
            xaxis=dict(tickmode='linear', dtick=1, gridcolor='#1e1e35', tickfont=dict(color='#5a5a82')),
        )
        st.plotly_chart(fig_area, use_container_width=True, config={'displayModeBar': False})

    with col6:
        source_counts = defaultdict(int)
        for h in history:
            for src in h.get('sources', []):
                source_counts[src] += 1

        if source_counts:
            top_sources = sorted(source_counts.items(), key=lambda x: x[1], reverse=True)[:8]
            src_labels = [s[0][:20] for s in top_sources]
            src_values = [s[1] for s in top_sources]

            fig_src = go.Figure(go.Bar(
                x=src_labels,
                y=src_values,
                marker=dict(
                    color=src_values,
                    colorscale=[[0, '#111120'], [0.5, '#5b8fff'], [1, '#39d98a']],
                    line=dict(color='rgba(0,0,0,0)', width=0)
                ),
                hovertemplate='<b>%{x}</b><br>Appearances: %{y}<extra></extra>',
            ))
            fig_src.update_layout(
                **PLOTLY_LAYOUT,
                title='Top News Sources',
                xaxis_title='Source',
                yaxis_title='Frequency',
                xaxis=dict(tickangle=-35, tickfont=dict(size=9, color='#5a5a82')),
                bargap=0.25,
            )
            st.plotly_chart(fig_src, use_container_width=True, config={'displayModeBar': False})
        else:
            st.markdown("""
            <div style="background:#0d0d18; border:1px solid #1e1e35; border-radius:8px; padding:24px; text-align:center; height:200px; display:flex; align-items:center; justify-content:center;">
                <div style="color:#5a5a82; font-size:0.8rem; font-family:'Fragment Mono',monospace;">Source data will appear here after more analyses</div>
            </div>
            """, unsafe_allow_html=True)

    # ---- ROW 4: Relationship breakdown stacked bar (if enough data) ----
    if len(history) >= 2:
        st.markdown('<div style="margin-top:8px;"></div>', unsafe_allow_html=True)

        rel_data = defaultdict(lambda: defaultdict(int))
        for i, h in enumerate(history):
            label = f"#{i+1}"
            for rel, cnt in h.get('relationship_counts', {}).items():
                rel_data[label][rel] += cnt

        if any(rel_data.values()):
            check_labels = [f"#{i+1}" for i in range(len(history))]
            rel_types = ['confirms', 'partially_confirms', 'contradicts', 'context_only', 'unrelated']
            rel_colors_map = {
                'confirms': '#39d98a',
                'partially_confirms': '#f5a623',
                'contradicts': '#ff4d6d',
                'context_only': '#5b8fff',
                'unrelated': '#1e1e35',
            }
            rel_display = {
                'confirms': 'CONFIRMS',
                'partially_confirms': 'PARTIAL',
                'contradicts': 'CONTRADICTS',
                'context_only': 'CONTEXT',
                'unrelated': 'UNRELATED',
            }

            fig_stack = go.Figure()
            for rel in rel_types:
                counts = [rel_data[lbl].get(rel, 0) for lbl in check_labels]
                if any(c > 0 for c in counts):
                    fig_stack.add_trace(go.Bar(
                        name=rel_display[rel],
                        x=check_labels,
                        y=counts,
                        marker_color=rel_colors_map[rel],
                        hovertemplate=f'<b>{rel_display[rel]}</b><br>Check: %{{x}}<br>Count: %{{y}}<extra></extra>',
                    ))

            fig_stack.update_layout(
                **PLOTLY_LAYOUT,
                title='Article Relationship Breakdown per Check',
                barmode='stack',
                xaxis_title='Check #',
                yaxis_title='Article Count',
                legend=dict(orientation='h', y=-0.25, font=dict(size=9, color='#5a5a82'), bgcolor='rgba(0,0,0,0)'),
                bargap=0.3,
            )
            st.plotly_chart(fig_stack, use_container_width=True, config={'displayModeBar': False})

    # ---- Summary Metrics Row ----
    st.markdown('<div style="margin-top:4px;"></div>', unsafe_allow_html=True)
    avg_conf = np.mean([h.get('confidence', 0) for h in history]) * 100 if history else 0
    misinfo_rate = sum(1 for h in history if h.get('verdict') in ['MISINFO', 'LIKELY_MISINFO']) / len(history) * 100 if history else 0
    authentic_rate = sum(1 for h in history if h.get('verdict') in ['AUTHENTIC', 'LIKELY_AUTHENTIC']) / len(history) * 100 if history else 0
    total_articles = sum(h.get('articles_fetched', 0) for h in history)

    m1, m2, m3, m4 = st.columns(4)
    for col, label, value, color in [
        (m1, "Avg Confidence", f"{avg_conf:.1f}%", "#5b8fff"),
        (m2, "Misinfo Rate", f"{misinfo_rate:.1f}%", "#ff4d6d"),
        (m3, "Authentic Rate", f"{authentic_rate:.1f}%", "#39d98a"),
        (m4, "Total Articles", str(total_articles), "#f5a623"),
    ]:
        with col:
            st.markdown(f"""
            <div style="background:#0d0d18; border:1px solid #1e1e35; border-radius:8px; padding:14px; text-align:center;">
                <div style="font-size:0.68rem; color:#5a5a82; font-family:'Fragment Mono',monospace; margin-bottom:6px; letter-spacing:0.08em;">{label}</div>
                <div style="font-family:'Space Grotesk',sans-serif; font-size:1.5rem; font-weight:800; color:{color};">{value}</div>
            </div>
            """, unsafe_allow_html=True)


def render_history_tab(history: list):
    """Render the History & Patterns tab."""

    st.markdown("""
    <div style="padding: 8px 0 24px 0;">
        <div style="font-family:'Fragment Mono',monospace; font-size:0.6rem; color:#5a5a82; letter-spacing:0.2em; text-transform:uppercase; margin-bottom:6px;">Session Memory</div>
        <h2 style="font-family:'Space Grotesk',sans-serif; font-size:1.7rem; font-weight:700; color:#d4d4f0; margin:0; letter-spacing:-0.03em;">
            History & Patterns
        </h2>
        <p style="color:#5a5a82; font-size:0.75rem; margin-top:6px; font-family:'Fragment Mono',monospace;">
            All claims checked this session · Pattern detection · Recurring topics
        </p>
    </div>
    """, unsafe_allow_html=True)

    if not history:
        st.markdown("""
        <div style="background:var(--surface); border:1px solid var(--border2); border-radius:4px; padding:48px; text-align:center; margin-top:20px; position:relative; overflow:hidden;">
            <div style="position:absolute; inset:0; background: radial-gradient(ellipse at center, rgba(57,217,138,0.03) 0%, transparent 70%); pointer-events:none;"></div>
            <div style="font-family:'Bebas Neue',sans-serif; font-size:3rem; color:rgba(57,217,138,0.1); letter-spacing:0.1em; margin-bottom:12px;">NO HISTORY</div>
            <div style="font-family:'Space Grotesk',sans-serif; font-weight:600; font-size:0.9rem; color:#3a3a58; margin-bottom:8px;">Analyze claims to build history</div>
            <div style="font-size:0.75rem; color:#2a2a48; font-family:'Fragment Mono',monospace; line-height:2;">
                Patterns will be detected automatically.
            </div>
        </div>
        """, unsafe_allow_html=True)
        return

    # ---- Quick Stats Strip ----
    total = len(history)
    misinfo_count = sum(1 for h in history if h.get('verdict') in ['MISINFO', 'LIKELY_MISINFO'])
    sensational_count = sum(1 for h in history if h.get('is_sensational', False))
    unique_types = len(set(h.get('claim_type', 'other') for h in history))

    s1, s2, s3, s4 = st.columns(4)
    for col, label, value, color in [
        (s1, "Total Checks", str(total), "#5b8fff"),
        (s2, "Misinfo Found", str(misinfo_count), "#ff4d6d"),
        (s3, "Sensational", str(sensational_count), "#f5a623"),
        (s4, "Unique Types", str(unique_types), "#39d98a"),
    ]:
        with col:
            st.markdown(f"""
            <div style="background:#0d0d18; border:1px solid #1e1e35; border-radius:8px; padding:12px; text-align:center; margin-bottom:16px;">
                <div style="font-size:0.65rem; color:#5a5a82; font-family:'Fragment Mono',monospace; margin-bottom:4px; letter-spacing:0.08em;">{label}</div>
                <div style="font-family:'Space Grotesk',sans-serif; font-size:1.6rem; font-weight:800; color:{color};">{value}</div>
            </div>
            """, unsafe_allow_html=True)

    # ---- Pattern Detection ----
    st.markdown('<div style="font-family:\'Syne\',sans-serif; font-weight:700; font-size:0.9rem; color:#5a5a82; margin: 4px 0 12px 0; letter-spacing:0.05em;">◈ DETECTED PATTERNS</div>', unsafe_allow_html=True)

    patterns = []

    if misinfo_count / total >= 0.5:
        patterns.append(("⚠", "#ff4d6d", f"High misinfo session — {misinfo_count}/{total} claims flagged as misinformation or likely misinfo."))
    if sensational_count / total >= 0.5:
        patterns.append(("⚡", "#f5a623", f"Sensational pattern detected — {sensational_count}/{total} claims used sensational/clickbait language."))

    type_counts = defaultdict(int)
    for h in history: type_counts[h.get('claim_type', 'other')] += 1
    dominant_type = max(type_counts, key=type_counts.get)
    if type_counts[dominant_type] / total >= 0.4:
        patterns.append(("◈", "#5b8fff", f"Topic cluster: Most claims ({type_counts[dominant_type]}/{total}) fall under '{dominant_type.upper()}' category."))

    confidences = [h.get('confidence', 0) for h in history]
    if len(confidences) >= 3:
        trend = confidences[-1] - confidences[0]
        if trend > 0.15:
            patterns.append(("↗", "#39d98a", "Confidence trend: Analysis confidence has been improving across checks."))
        elif trend < -0.15:
            patterns.append(("↘", "#f5a623", "Confidence trend: Analysis confidence has been declining — claims may be more ambiguous."))

    avg_articles = np.mean([h.get('articles_fetched', 0) for h in history])
    if avg_articles < 5:
        patterns.append(("📡", "#5a5a82", f"Low article coverage: Averaging only {avg_articles:.1f} articles per check. Try more specific claims."))

    if not patterns:
        patterns.append(("✓", "#39d98a", "No strong patterns detected yet. Keep analyzing more claims for pattern insights."))

    for icon, color, text in patterns:
        st.markdown(f"""
        <div style="background:#0d0d18; border:1px solid #1e1e35; border-left:3px solid {color}; border-radius:0 8px 8px 0; padding:12px 16px; margin-bottom:8px; display:flex; gap:12px; align-items:flex-start;">
            <span style="color:{color}; font-size:1.1rem; margin-top:1px;">{icon}</span>
            <span style="font-size:0.82rem; color:#9090b8; font-family:'Fragment Mono',monospace; line-height:1.6;">{text}</span>
        </div>
        """, unsafe_allow_html=True)

    # ---- All Past Claims ----
    st.markdown('<div style="font-family:\'Syne\',sans-serif; font-weight:700; font-size:0.9rem; color:#5a5a82; margin: 20px 0 12px 0; letter-spacing:0.05em;">◈ ALL CHECKED CLAIMS</div>', unsafe_allow_html=True)

    verdict_config = {
        'AUTHENTIC':          ('✅', '#39d98a', 'AUTHENTIC'),
        'LIKELY_AUTHENTIC':   ('✔',  '#39d98a', 'LIKELY AUTH'),
        'UNVERIFIED':         ('◈',  '#5b8fff', 'UNVERIFIED'),
        'LIKELY_MISINFO':     ('⚠',  '#f5a623', 'LIKELY MISINFO'),
        'MISINFO':            ('✗',  '#ff4d6d', 'MISINFO'),
        'INSUFFICIENT_DATA':  ('?',  '#5a5a82', 'INSUFF. DATA'),
    }

    for i, h in enumerate(reversed(history)):
        idx = total - i
        verdict = h.get('verdict', 'INSUFFICIENT_DATA')
        icon, color, label = verdict_config.get(verdict, ('?', '#5a5a82', verdict))
        conf = h.get('confidence', 0)
        claim_type = h.get('claim_type', 'other').upper()
        ts = h.get('timestamp', '')
        articles_n = h.get('articles_fetched', 0)
        sensational = h.get('is_sensational', False)
        claim_text = h.get('claim', '')[:120]
        if len(h.get('claim', '')) > 120: claim_text += '…'

        sensational_badge = f'<span style="background:rgba(255,181,71,0.15);color:#f5a623;border:1px solid rgba(255,181,71,0.3);font-size:0.65rem;padding:1px 7px;border-radius:10px;font-family:Fragment Mono,monospace;margin-left:6px;">⚡ SENSATIONAL</span>' if sensational else ''

        st.markdown(f"""
        <div style="background:#0d0d18; border:1px solid #1e1e35; border-radius:8px; padding:14px 18px; margin-bottom:8px;">
            <div style="display:flex; justify-content:space-between; align-items:flex-start; margin-bottom:8px;">
                <div style="flex:1; padding-right:16px;">
                    <div style="font-size:0.68rem; color:#5a5a82; font-family:'Fragment Mono',monospace; margin-bottom:4px; letter-spacing:0.05em;">
                        CHECK #{idx} · {ts} · {claim_type} · {articles_n} articles
                        {sensational_badge}
                    </div>
                    <div style="font-family:'Space Grotesk',sans-serif; font-weight:600; font-size:0.88rem; color:#d4d4f0; line-height:1.5;">{claim_text}</div>
                </div>
                <div style="text-align:right; min-width:90px;">
                    <div style="font-size:1.1rem; margin-bottom:2px;">{icon}</div>
                    <div style="font-family:'Space Grotesk',sans-serif; font-weight:700; font-size:0.75rem; color:{color};">{label}</div>
                    <div style="font-size:0.7rem; color:#5a5a82; font-family:'Fragment Mono',monospace;">{conf:.0%} conf</div>
                </div>
            </div>
            <div style="background:#050508; border-radius:2px; height:2px;">
                <div style="background:{color}; width:{int(conf*100)}%; height:2px; border-radius:2px; opacity:0.6;"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ---- Topic Frequency Map ----
    all_topics = []
    for h in history:
        all_topics.extend(h.get('key_topics', []))

    if all_topics:
        topic_freq = defaultdict(int)
        for t in all_topics: topic_freq[t.lower()] += 1
        top_topics = sorted(topic_freq.items(), key=lambda x: x[1], reverse=True)[:15]

        st.markdown('<div style="font-family:\'Syne\',sans-serif; font-weight:700; font-size:0.9rem; color:#5a5a82; margin: 20px 0 12px 0; letter-spacing:0.05em;">◈ RECURRING TOPICS THIS SESSION</div>', unsafe_allow_html=True)

        topic_html = ""
        for topic, freq in top_topics:
            opacity = 0.4 + (freq / top_topics[0][1]) * 0.6
            size = 0.72 + (freq / top_topics[0][1]) * 0.3
            topic_html += f'<span style="display:inline-block; background:rgba(0,229,255,{opacity*0.15:.2f}); border:1px solid rgba(0,229,255,{opacity*0.4:.2f}); color:rgba(0,229,255,{opacity:.2f}); padding:4px 12px; border-radius:20px; font-size:{size:.2f}rem; margin:4px; font-family:Fragment Mono,monospace; font-weight:600;">{topic} <span style="opacity:0.5; font-size:0.65rem;">×{freq}</span></span>'

        st.markdown(f'<div style="background:#0d0d18; border:1px solid #1e1e35; border-radius:8px; padding:16px; line-height:2.2;">{topic_html}</div>', unsafe_allow_html=True)

    # ---- Clear History Button ----
    st.markdown('<div style="margin-top:24px;"></div>', unsafe_allow_html=True)
    if st.button("🗑 Clear History", use_container_width=False):
        st.session_state.analysis_history = []
        st.rerun()


# ============================================
# STREAMLIT UI
# ============================================

st.set_page_config(
    page_title="TruthLens — Misinformation Tracker",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=Fragment+Mono:ital@0;1&family=Bebas+Neue&display=swap');

:root {
    --bg: #050508;
    --bg2: #08080e;
    --surface: #0d0d18;
    --surface2: #111120;
    --surface3: #161628;
    --border: #1e1e35;
    --border2: #2a2a48;
    --accent: #5b8fff;
    --accent-glow: rgba(91,143,255,0.18);
    --accent2: #ff4d6d;
    --accent2-glow: rgba(255,77,109,0.15);
    --accent3: #39d98a;
    --accent3-glow: rgba(57,217,138,0.15);
    --warn: #f5a623;
    --warn-glow: rgba(245,166,35,0.15);
    --text: #d4d4f0;
    --text-dim: #5a5a82;
    --text-mid: #9090b8;
    --pulse: #5b8fff;
}

/* ===== BASE ===== */
html, body, .stApp {
    background: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'Space Grotesk', sans-serif !important;
}
h1, h2, h3 { font-family: 'Space Grotesk', sans-serif !important; font-weight: 700 !important; }

/* Animated noise overlay */
.stApp::before {
    content: '';
    position: fixed;
    inset: 0;
    background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='noise'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23noise)' opacity='0.03'/%3E%3C/svg%3E");
    pointer-events: none;
    z-index: 0;
    opacity: 0.4;
}

/* Scanline effect */
.stApp::after {
    content: '';
    position: fixed;
    inset: 0;
    background: repeating-linear-gradient(
        0deg,
        transparent,
        transparent 2px,
        rgba(0,0,0,0.03) 2px,
        rgba(0,0,0,0.03) 4px
    );
    pointer-events: none;
    z-index: 1;
}

/* ===== SIDEBAR ===== */
[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border2) !important;
}
[data-testid="stSidebar"]::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent, var(--accent), transparent);
    opacity: 0.6;
}

/* ===== INPUTS ===== */
.stTextArea textarea, .stTextInput input {
    background: var(--surface2) !important;
    border: 1px solid var(--border2) !important;
    color: var(--text) !important;
    font-family: 'Fragment Mono', monospace !important;
    font-size: 0.875rem !important;
    border-radius: 4px !important;
    transition: all 0.25s !important;
}
.stTextArea textarea:focus, .stTextInput input:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 1px var(--accent), 0 0 20px var(--accent-glow) !important;
    background: var(--surface3) !important;
}

/* ===== BUTTONS ===== */
.stButton > button {
    background: var(--surface2) !important;
    border: 1px solid var(--border2) !important;
    color: var(--text-mid) !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 0.8rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    border-radius: 3px !important;
    padding: 8px 20px !important;
    transition: all 0.2s !important;
}
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #3a6bff 0%, #6b4fff 100%) !important;
    color: #fff !important;
    border: none !important;
    box-shadow: 0 0 24px rgba(91,143,255,0.35), inset 0 1px 0 rgba(255,255,255,0.15) !important;
    font-size: 0.85rem !important;
    letter-spacing: 0.12em !important;
    padding: 10px 28px !important;
}
.stButton > button[kind="primary"]:hover {
    box-shadow: 0 0 40px rgba(91,143,255,0.55), inset 0 1px 0 rgba(255,255,255,0.2) !important;
    transform: translateY(-1px) !important;
}
.stButton > button:hover {
    border-color: var(--accent) !important;
    color: var(--text) !important;
    background: var(--surface3) !important;
}

/* ===== METRICS ===== */
[data-testid="stMetric"] {
    background: var(--surface2) !important;
    border: 1px solid var(--border2) !important;
    border-radius: 4px !important;
    padding: 16px !important;
    position: relative !important;
    overflow: hidden !important;
}
[data-testid="stMetric"]::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, var(--accent), #a64fff);
}
[data-testid="stMetricValue"] {
    color: var(--accent) !important;
    font-family: 'Bebas Neue', sans-serif !important;
    font-size: 2rem !important;
    letter-spacing: 0.04em !important;
}
[data-testid="stMetricLabel"] { color: var(--text-dim) !important; font-size: 0.7rem !important; letter-spacing: 0.1em !important; text-transform: uppercase !important; }

/* ===== EXPANDERS ===== */
[data-testid="stExpander"] {
    background: var(--surface) !important;
    border: 1px solid var(--border2) !important;
    border-radius: 4px !important;
}
[data-testid="stExpander"]:hover { border-color: var(--accent) !important; }

/* ===== STATUS/ALERTS ===== */
.stSuccess { background: rgba(57,217,138,0.06) !important; border: 1px solid rgba(57,217,138,0.25) !important; border-radius: 4px !important; }
.stWarning { background: rgba(245,166,35,0.06) !important; border: 1px solid rgba(245,166,35,0.25) !important; border-radius: 4px !important; }
.stError { background: rgba(255,77,109,0.06) !important; border: 1px solid rgba(255,77,109,0.25) !important; border-radius: 4px !important; }
.stInfo { background: rgba(91,143,255,0.05) !important; border: 1px solid rgba(91,143,255,0.2) !important; border-radius: 4px !important; }
.stProgress > div > div { background: linear-gradient(90deg, var(--accent), #a64fff) !important; }
hr { border-color: var(--border) !important; }
.stCaption { color: var(--text-dim) !important; font-size: 0.72rem !important; }

/* ===== TABS ===== */
[data-testid="stTabs"] [data-baseweb="tab-list"] {
    background: transparent !important;
    border-bottom: 1px solid var(--border2) !important;
    gap: 0 !important;
    padding: 0 !important;
}
[data-testid="stTabs"] [data-baseweb="tab"] {
    background: transparent !important;
    border: none !important;
    border-bottom: 2px solid transparent !important;
    color: var(--text-dim) !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 0.78rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    padding: 14px 24px !important;
    transition: all 0.2s !important;
}
[data-testid="stTabs"] [aria-selected="true"] {
    color: var(--accent) !important;
    border-bottom: 2px solid var(--accent) !important;
    background: rgba(91,143,255,0.04) !important;
}
[data-testid="stTabs"] [data-baseweb="tab"]:hover {
    color: var(--text) !important;
    background: rgba(91,143,255,0.03) !important;
}
[data-testid="stTabPanel"] {
    background: transparent !important;
    padding-top: 24px !important;
}

/* ===== VERDICT CARDS ===== */
.verdict-authentic {
    background: linear-gradient(135deg, rgba(57,217,138,0.06) 0%, rgba(57,217,138,0.02) 100%);
    border: 1px solid rgba(57,217,138,0.4);
    border-left: 4px solid #39d98a;
    border-radius: 4px;
    padding: 24px 28px;
    margin: 16px 0;
    position: relative;
    overflow: hidden;
}
.verdict-authentic::before {
    content: 'AUTHENTIC';
    position: absolute;
    top: 50%; right: 24px;
    transform: translateY(-50%);
    font-family: 'Bebas Neue', sans-serif;
    font-size: 4rem;
    color: rgba(57,217,138,0.06);
    letter-spacing: 0.1em;
    pointer-events: none;
    white-space: nowrap;
}
.verdict-misinfo {
    background: linear-gradient(135deg, rgba(255,77,109,0.07) 0%, rgba(255,77,109,0.02) 100%);
    border: 1px solid rgba(255,77,109,0.4);
    border-left: 4px solid #ff4d6d;
    border-radius: 4px;
    padding: 24px 28px;
    margin: 16px 0;
    position: relative;
    overflow: hidden;
}
.verdict-misinfo::before {
    content: 'FALSE';
    position: absolute;
    top: 50%; right: 24px;
    transform: translateY(-50%);
    font-family: 'Bebas Neue', sans-serif;
    font-size: 4rem;
    color: rgba(255,77,109,0.06);
    letter-spacing: 0.1em;
    pointer-events: none;
}
.verdict-warning {
    background: linear-gradient(135deg, rgba(245,166,35,0.07) 0%, rgba(245,166,35,0.02) 100%);
    border: 1px solid rgba(245,166,35,0.4);
    border-left: 4px solid #f5a623;
    border-radius: 4px;
    padding: 24px 28px;
    margin: 16px 0;
    position: relative;
    overflow: hidden;
}
.verdict-warning::before {
    content: 'CAUTION';
    position: absolute;
    top: 50%; right: 24px;
    transform: translateY(-50%);
    font-family: 'Bebas Neue', sans-serif;
    font-size: 4rem;
    color: rgba(245,166,35,0.06);
    letter-spacing: 0.1em;
    pointer-events: none;
}
.verdict-neutral {
    background: linear-gradient(135deg, rgba(91,143,255,0.06) 0%, rgba(91,143,255,0.02) 100%);
    border: 1px solid rgba(91,143,255,0.35);
    border-left: 4px solid #5b8fff;
    border-radius: 4px;
    padding: 24px 28px;
    margin: 16px 0;
    position: relative;
    overflow: hidden;
}
.verdict-neutral::before {
    content: 'UNCLEAR';
    position: absolute;
    top: 50%; right: 24px;
    transform: translateY(-50%);
    font-family: 'Bebas Neue', sans-serif;
    font-size: 4rem;
    color: rgba(91,143,255,0.06);
    letter-spacing: 0.1em;
    pointer-events: none;
}

/* ===== ARTICLE CARDS ===== */
.article-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 16px 20px;
    margin: 6px 0;
    transition: all 0.2s;
    position: relative;
}
.article-card::before {
    content: '';
    position: absolute;
    left: 0; top: 0; bottom: 0;
    width: 2px;
    background: var(--border2);
    transition: all 0.2s;
}
.article-card:hover {
    border-color: var(--border2);
    background: var(--surface2);
}
.article-card:hover::before {
    background: var(--accent);
}

/* ===== TAGS ===== */
.tag {
    display: inline-block;
    padding: 2px 9px;
    border-radius: 2px;
    font-size: 0.65rem;
    font-weight: 700;
    margin: 2px;
    font-family: 'Fragment Mono', monospace;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}
.tag-confirms { background: rgba(57,217,138,0.12); color: #39d98a; border: 1px solid rgba(57,217,138,0.3); }
.tag-contradicts { background: rgba(255,77,109,0.12); color: #ff4d6d; border: 1px solid rgba(255,77,109,0.3); }
.tag-partial { background: rgba(245,166,35,0.12); color: #f5a623; border: 1px solid rgba(245,166,35,0.3); }
.tag-unrelated { background: rgba(90,90,130,0.12); color: #5a5a82; border: 1px solid rgba(90,90,130,0.3); }
.tag-context { background: rgba(91,143,255,0.1); color: #5b8fff; border: 1px solid rgba(91,143,255,0.25); }

/* ===== STAT ROWS ===== */
.stat-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 7px 0;
    border-bottom: 1px solid var(--border);
    font-size: 0.75rem;
}
.stat-label { color: var(--text-dim); font-family: 'Fragment Mono', monospace; font-size: 0.7rem; }
.stat-value { color: var(--accent); font-weight: 700; font-family: 'Fragment Mono', monospace; }

/* ===== ERROR BOX ===== */
.error-box {
    background: rgba(255,77,109,0.06);
    border: 1px solid rgba(255,77,109,0.35);
    border-left: 3px solid #ff4d6d;
    border-radius: 4px;
    padding: 16px 20px;
    margin: 12px 0;
    font-size: 0.82rem;
    font-family: 'Fragment Mono', monospace;
    color: #ff4d6d;
    line-height: 1.8;
}

/* ===== INTEL BADGE ===== */
.intel-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: var(--surface2);
    border: 1px solid var(--border2);
    border-radius: 2px;
    padding: 3px 10px;
    font-family: 'Fragment Mono', monospace;
    font-size: 0.65rem;
    color: var(--text-dim);
    letter-spacing: 0.08em;
    text-transform: uppercase;
}
.intel-badge .dot {
    width: 5px; height: 5px;
    border-radius: 50%;
    background: var(--accent3);
    box-shadow: 0 0 6px var(--accent3);
    animation: pulse 2s infinite;
}
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.3; }
}

/* ===== SIDEBAR SECTION HEADER ===== */
.sb-section {
    font-family: 'Fragment Mono', monospace;
    font-size: 0.62rem;
    color: var(--text-dim);
    letter-spacing: 0.2em;
    text-transform: uppercase;
    padding: 4px 0 8px 0;
    border-bottom: 1px solid var(--border);
    margin-bottom: 12px;
}

/* ===== INFO PANEL ===== */
.info-panel {
    background: var(--surface);
    border: 1px solid var(--border2);
    border-radius: 4px;
    padding: 16px;
    height: 100%;
}

/* ===== INTEL GRID ===== */
.intel-grid {
    display: grid;
    grid-template-columns: 1fr 1fr 1fr;
    gap: 1px;
    background: var(--border);
    border: 1px solid var(--border);
    border-radius: 4px;
    overflow: hidden;
    margin: 12px 0;
}
.intel-cell {
    background: var(--surface);
    padding: 12px 14px;
}
.intel-cell-label {
    font-family: 'Fragment Mono', monospace;
    font-size: 0.6rem;
    color: var(--text-dim);
    letter-spacing: 0.15em;
    text-transform: uppercase;
    margin-bottom: 4px;
}
.intel-cell-value {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 0.9rem;
    font-weight: 700;
    color: var(--text);
}
</style>
""", unsafe_allow_html=True)

# ---- HEADER ----
st.markdown("""
<div style="padding: 20px 0 28px 0; position: relative;">
    <div style="display:flex; align-items:center; gap:10px; margin-bottom:10px;">
        <span class="intel-badge"><span class="dot"></span>SYSTEM ONLINE</span>
        <span class="intel-badge">TRUTHLENS v2.0</span>
        <span class="intel-badge">GROQ LLAMA 3.3</span>
    </div>
    <h1 style="font-family:'Space Grotesk',sans-serif; font-size:2.6rem; font-weight:700; color:#d4d4f0; margin:0 0 4px 0; letter-spacing:-0.04em; line-height:1.1;">
        Misinformation
        <span style="background: linear-gradient(135deg, #5b8fff, #a64fff); -webkit-background-clip:text; -webkit-text-fill-color:transparent; background-clip:text;">Intelligence</span>
        Platform
    </h1>
    <p style="color:#5a5a82; font-size:0.8rem; margin-top:8px; font-family:'Fragment Mono',monospace; letter-spacing:0.04em;">
        Real-time semantic fact-checking · 40+ data structures · NewsAPI · Completely Free
    </p>
    <div style="position:absolute; top:20px; right:0; width:280px; height:60px; background: radial-gradient(ellipse at right, rgba(91,143,255,0.08) 0%, transparent 70%); pointer-events:none;"></div>
</div>
""", unsafe_allow_html=True)

# ---- SESSION STATE ----
if 'tracker' not in st.session_state:
    st.session_state.tracker = MisinformationTracker()
if 'analysis_result' not in st.session_state:
    st.session_state.analysis_result = None
if 'query_meta' not in st.session_state:
    st.session_state.query_meta = None
if 'error_message' not in st.session_state:
    st.session_state.error_message = None
if 'analysis_history' not in st.session_state:
    st.session_state.analysis_history = []
if 'current_claim' not in st.session_state:
    st.session_state.current_claim = ''

tracker = st.session_state.tracker

# ---- LOAD API KEYS FROM ENV / st.secrets ----
def get_api_keys():
    # Try st.secrets first (Streamlit Cloud), then .env / environment variables
    groq_key = ""
    news_key = ""
    try:
        groq_key = st.secrets.get("GROQ_API_KEY", "")
        news_key = st.secrets.get("NEWS_API_KEY", "")
    except Exception:
        pass
    if not groq_key:
        groq_key = os.environ.get("GROQ_API_KEY", "")
    if not news_key:
        news_key = os.environ.get("NEWS_API_KEY", "")
    return groq_key, news_key

groq_api_key, news_api_key = get_api_keys()

# ---- SIDEBAR ----
with st.sidebar:
    st.markdown("""
    <div style="padding: 20px 0 16px 0;">
        <div style="font-family:'Fragment Mono',monospace; font-size:0.6rem; color:#5a5a82; letter-spacing:0.2em; margin-bottom:6px;">CONTROL PANEL</div>
        <div style="font-family:'Space Grotesk',sans-serif; font-size:1.1rem; font-weight:700; color:#d4d4f0; letter-spacing:-0.02em;">TruthLens</div>
    </div>
    """, unsafe_allow_html=True)

    if not GROQ_AVAILABLE:
        st.error("⚠ Groq not installed. Run: `pip install groq` then restart.")

    if not groq_api_key or not news_api_key:
        st.warning("⚠ API keys not found. Add them to your `.env` file.", icon="🔑")

    st.markdown("---")
    st.markdown('<div class="sb-section">Index Statistics</div>', unsafe_allow_html=True)

    stats = tracker.get_statistics()
    for key, val in stats.items():
        st.markdown(f"""
        <div class="stat-row">
            <span class="stat-label">{key}</span>
            <span class="stat-value">{val}</span>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    # Session History Quick View in Sidebar
    if st.session_state.analysis_history:
        history_count = len(st.session_state.analysis_history)
        misinfo_n = sum(1 for h in st.session_state.analysis_history if h.get('verdict') in ['MISINFO', 'LIKELY_MISINFO'])
        st.markdown(f"""
        <div style="font-family:'Fragment Mono',monospace; font-size:0.62rem; letter-spacing:0.2em; text-transform:uppercase; color:#5a5a82; margin-bottom:10px;">Session</div>
        <div class="stat-row"><span class="stat-label">Checks run</span><span class="stat-value">{history_count}</span></div>
        <div class="stat-row"><span class="stat-label">Misinfo found</span><span class="stat-value" style="color:#ff4d6d;">{misinfo_n}</span></div>
        <div style="margin-bottom:12px;"></div>
        """, unsafe_allow_html=True)

    if st.button("🗑 Reset Tracker", use_container_width=True):
        st.session_state.tracker = MisinformationTracker()
        st.session_state.analysis_result = None
        st.session_state.query_meta = None
        st.session_state.error_message = None
        st.rerun()

# ---- TABS ----
tab_main, tab_viz, tab_history = st.tabs([
    "⬡  Analyze Claim",
    "◈  Intelligence Dashboard",
    "▤  History & Patterns"
])

# ============================================
# TAB 1: MAIN ANALYSIS (original app, unchanged)
# ============================================
with tab_main:

    col_input, col_guide = st.columns([3, 1])

    with col_input:
        news_query = st.text_area(
            "Enter any news claim to verify",
            placeholder="Examples:\n• Scientists found a cure for cancer in 2025\n• People are dying due to corona virus\n• Government is hiding UFO evidence\n• Stock market crashed today",
            height=130,
            key="main_query"
        )

    with col_guide:
        st.markdown("""
        <div class="info-panel">
            <div style="font-family:'Fragment Mono',monospace; font-size:0.6rem; color:#5a5a82; letter-spacing:0.18em; text-transform:uppercase; margin-bottom:12px;">How It Works</div>
            <div style="font-size:0.72rem; color:#5a5a82; line-height:2.1; font-family:'Fragment Mono',monospace;">
                <span style="color:#5b8fff; font-weight:700;">01</span> &nbsp;Groq extracts intent<br>
                <span style="color:#5b8fff; font-weight:700;">02</span> &nbsp;Multi-query news search<br>
                <span style="color:#5b8fff; font-weight:700;">03</span> &nbsp;40+ structures index data<br>
                <span style="color:#5b8fff; font-weight:700;">04</span> &nbsp;Groq semantic scoring<br>
                <span style="color:#5b8fff; font-weight:700;">05</span> &nbsp;Reasoned verdict
            </div>
        </div>
        """, unsafe_allow_html=True)

    analyze_btn = st.button("🔬 Analyze Claim", type="primary")

    # ---- ANALYSIS LOGIC ----
    if analyze_btn and news_query.strip():

        if not GROQ_AVAILABLE:
            st.error("❌ Please install groq first: `pip install groq` then restart the app.")
            st.stop()

        if not groq_api_key:
            st.error("⚠️ Groq API key required. Get your free key at console.groq.com")
            st.stop()

        if not news_api_key:
            st.error("⚠️ NewsAPI key required. Get your free key at newsapi.org/register")
            st.stop()

        st.session_state.tracker = MisinformationTracker()
        tracker = st.session_state.tracker
        st.session_state.analysis_result = None
        st.session_state.query_meta = None
        st.session_state.error_message = None
        st.session_state.current_claim = news_query

        try:
            with st.status("🧠 **Phase 1:** Groq analyzing claim intent...", expanded=True) as status:
                st.write("Extracting key topics and generating optimized search queries...")
                query_meta = expand_query_with_groq(news_query, groq_api_key)
                st.session_state.query_meta = query_meta
                st.write(f"✓ Claim type: **{query_meta.get('claim_type', 'unknown').upper()}**")
                st.write(f"✓ Primary search query: **\"{query_meta.get('primary_query', '')}\"**")
                st.write(f"✓ Secondary searches: {query_meta.get('secondary_queries', [])}")
                if query_meta.get('is_sensational'):
                    st.write("⚡ **Sensational language detected**")
                status.update(label="✅ Phase 1 complete — Search queries generated", state="complete")

            with st.status("📡 **Phase 2:** Fetching real news articles...", expanded=True) as status:
                queries = [query_meta.get('primary_query', news_query)] + query_meta.get('secondary_queries', [])
                st.write(f"Searching {len(queries)} query variations...")

                articles = search_news_multi_query(queries, news_api_key, page_size=15)

                if not articles:
                    status.update(label="⚠️ Phase 2 — No articles found for these queries", state="error")
                    st.session_state.error_message = (
                        f"No articles were found for the queries: {queries}. "
                        "Try rephrasing your claim with more specific keywords."
                    )
                    st.rerun()

                for article in articles:
                    tracker.add_article(
                        title=article['title'],
                        description=article.get('description', ''),
                        source=article['source'],
                        category=article.get('category', 'news'),
                        url=article.get('url', ''),
                        published_at=article.get('published_at', '')
                    )

                tracker.last_update = datetime.now()
                unique_sources = len(set(a['source'] for a in articles))
                st.write(f"✓ Fetched **{len(articles)} unique articles** from **{unique_sources} sources**")
                st.write("✓ Indexed into AVL Tree, B+ Tree, Trie, BK-Tree, Skip List, Union-Find...")
                status.update(label=f"✅ Phase 2 complete — {len(articles)} articles indexed", state="complete")

            with st.status("🔬 **Phase 3:** Groq performing semantic fact-check...", expanded=True) as status:
                st.write("Scoring each article's relevance and relationship to claim...")
                analysis = analyze_articles_with_groq(news_query, tracker.articles, groq_api_key)
                st.session_state.analysis_result = analysis

                scored_count = len(analysis.get('article_scores', []))
                st.write(f"✓ Semantically scored **{scored_count} relevant articles**")
                st.write(f"✓ Red flags detected: {len(analysis.get('red_flags', []))}")
                st.write(f"✓ Supporting evidence found: {len(analysis.get('supporting_evidence', []))}")

                version = tracker.persistent.update(0, news_query)
                st.write(f"✓ Query stored in Persistent Array (Version {version})")
                status.update(label="✅ Phase 3 complete — Analysis ready", state="complete")

            # ---- Save to history ----
            rel_counts = defaultdict(int)
            for score_data in analysis.get('article_scores', []):
                rel_counts[score_data.get('relationship', 'unrelated')] += 1

            sources_list = list(set(a['source'] for a in tracker.articles))

            st.session_state.analysis_history.append({
                'claim': news_query,
                'verdict': analysis.get('verdict', 'INSUFFICIENT_DATA'),
                'confidence': analysis.get('confidence', 0.0),
                'claim_type': query_meta.get('claim_type', 'other'),
                'is_sensational': query_meta.get('is_sensational', False),
                'key_topics': query_meta.get('key_topics', []),
                'articles_fetched': len(tracker.articles),
                'sources': sources_list,
                'relationship_counts': dict(rel_counts),
                'red_flags_count': len(analysis.get('red_flags', [])),
                'evidence_count': len(analysis.get('supporting_evidence', [])),
                'timestamp': datetime.now().strftime("%H:%M:%S"),
            })

            st.rerun()

        except Exception as e:
            st.session_state.error_message = str(e)
            st.rerun()

    # ---- ERROR DISPLAY ----
    if st.session_state.error_message:
        st.markdown(f"""
        <div class="error-box">
            <div style="font-family:'Space Grotesk',sans-serif; font-weight:700; font-size:0.85rem; margin-bottom:6px; letter-spacing:0.05em;">⚠ ANALYSIS ERROR</div>
            {st.session_state.error_message}
        </div>
        """, unsafe_allow_html=True)

    # ---- RESULTS DISPLAY ----
    if st.session_state.analysis_result and st.session_state.query_meta:
        analysis = st.session_state.analysis_result
        query_meta = st.session_state.query_meta
        verdict = analysis.get('verdict', 'INSUFFICIENT_DATA')
        confidence = analysis.get('confidence', 0.0)

        st.markdown("---")
        st.markdown('<div style="font-family:\'Fragment Mono\',monospace; font-size:0.62rem; color:#5a5a82; letter-spacing:0.2em; text-transform:uppercase; margin-bottom:16px;">Analysis Results</div>', unsafe_allow_html=True)

        verdict_config = {
            'AUTHENTIC':          ('verdict-authentic', '✅', '#39d98a', 'AUTHENTIC'),
            'LIKELY_AUTHENTIC':   ('verdict-authentic', '✔',  '#39d98a', 'LIKELY AUTHENTIC'),
            'UNVERIFIED':         ('verdict-neutral',   '◈',  '#5b8fff', 'UNVERIFIED'),
            'LIKELY_MISINFO':     ('verdict-warning',   '⚠',  '#f5a623', 'LIKELY MISINFORMATION'),
            'MISINFO':            ('verdict-misinfo',   '✗',  '#ff4d6d', 'MISINFORMATION'),
            'INSUFFICIENT_DATA':  ('verdict-neutral',   '?',  '#5b8fff', 'INSUFFICIENT DATA'),
        }
        css_class, icon, color, label = verdict_config.get(verdict, ('verdict-neutral', '?', '#5b8fff', verdict))
        sensational_badge = ' · ⚡ SENSATIONAL CLAIM DETECTED' if query_meta.get('is_sensational') else ''

        st.markdown(f"""
        <div class="{css_class}">
            <div style="display:flex; align-items:center; gap:14px; margin-bottom:12px;">
                <span style="font-family:'Bebas Neue',sans-serif; font-size:3rem; color:{color}; line-height:1; letter-spacing:0.04em;">{label}</span>
            </div>
            <div style="font-family:'Fragment Mono',monospace; font-size:0.72rem; color:{color}; opacity:0.8; margin-bottom:12px; letter-spacing:0.06em;">
                CONFIDENCE {confidence:.0%} &nbsp;·&nbsp; {query_meta.get('claim_type','unknown').upper()}{sensational_badge}
            </div>
            <p style="color:#9090b8; font-size:0.85rem; line-height:1.8; margin:0 0 12px 0; font-family:'Space Grotesk',sans-serif;">
                {analysis.get('verdict_reasoning', '')}
            </p>
            <div style="font-size:0.75rem; color:#5a5a82; font-family:'Fragment Mono',monospace; border-top:1px solid rgba(255,255,255,0.06); padding-top:10px; margin-top:8px; letter-spacing:0.02em;">
                → {analysis.get('recommended_action', '')}
            </div>
        </div>
        """, unsafe_allow_html=True)

        if analysis.get('fact_check_summary'):
            with st.expander("▤ Full Fact-Check Summary", expanded=True):
                st.markdown(f"""
                <div style="font-family:'Fragment Mono',monospace; font-size:0.82rem; line-height:1.9; color:#9090b8;">
                    {analysis['fact_check_summary']}
                </div>
                """, unsafe_allow_html=True)

        col_rf, col_ev = st.columns(2)
        with col_rf:
            red_flags = analysis.get('red_flags', [])
            if red_flags:
                items_html = ''.join(
                    f'<div style="font-size:0.76rem; color:#9090b8; padding:5px 0; border-bottom:1px solid #111120; font-family:Fragment Mono,monospace; line-height:1.6;">— {flag}</div>'
                    for flag in red_flags
                )
                st.markdown(f"""
                <div style="background:rgba(255,77,109,0.04); border:1px solid rgba(255,77,109,0.2); border-left:3px solid #ff4d6d; border-radius:4px; padding:16px;">
                    <div style="font-family:'Fragment Mono',monospace; font-weight:700; color:#ff4d6d; margin-bottom:10px; font-size:0.65rem; letter-spacing:0.15em; text-transform:uppercase;">⚠ Red Flags ({len(red_flags)})</div>
                    {items_html}
                </div>
                """, unsafe_allow_html=True)

        with col_ev:
            evidence = analysis.get('supporting_evidence', [])
            if evidence:
                items_html = ''.join(
                    f'<div style="font-size:0.76rem; color:#9090b8; padding:5px 0; border-bottom:1px solid #111120; font-family:Fragment Mono,monospace; line-height:1.6;">— {ev}</div>'
                    for ev in evidence
                )
                st.markdown(f"""
                <div style="background:rgba(57,217,138,0.04); border:1px solid rgba(57,217,138,0.2); border-left:3px solid #39d98a; border-radius:4px; padding:16px;">
                    <div style="font-family:'Fragment Mono',monospace; font-weight:700; color:#39d98a; margin-bottom:10px; font-size:0.65rem; letter-spacing:0.15em; text-transform:uppercase;">✓ Supporting Evidence ({len(evidence)})</div>
                    {items_html}
                </div>
                """, unsafe_allow_html=True)

        if tracker.articles:
            st.markdown('<div style="font-family:\'Fragment Mono\',monospace; font-size:0.62rem; color:#5a5a82; letter-spacing:0.2em; text-transform:uppercase; margin:24px 0 12px 0;">Article Breakdown</div>', unsafe_allow_html=True)

            scored_articles = analysis.get('article_scores', [])
            scored_articles_sorted = sorted(scored_articles, key=lambda x: x.get('similarity_score', 0), reverse=True)

            relationship_tags = {
                'confirms':           'tag-confirms',
                'contradicts':        'tag-contradicts',
                'partially_confirms': 'tag-partial',
                'unrelated':          'tag-unrelated',
                'context_only':       'tag-context',
            }

            for score_data in scored_articles_sorted[:10]:
                idx = score_data.get('index', 0)
                sim = score_data.get('similarity_score', 0)
                rel = score_data.get('relationship', 'unrelated')
                explanation = score_data.get('relevance_explanation', '')

                if idx < len(tracker.articles):
                    article = tracker.articles[idx]
                    tag_class = relationship_tags.get(rel, 'tag-unrelated')
                    rel_label = rel.replace('_', ' ').upper()
                    bar_color = '#39d98a' if sim > 0.6 else '#f5a623' if sim > 0.3 else '#5a5a82'
                    bar_width = int(sim * 100)
                    pub_date = article.get('published_at', '')[:10]
                    read_link = (
                        f'<a href="{article["url"]}" target="_blank" style="font-size:0.68rem; color:#5b8fff; text-decoration:none; font-family:Fragment Mono,monospace; letter-spacing:0.04em;">↗ Read article</a>'
                        if article.get("url") else ''
                    )

                    st.markdown(f"""
                    <div class="article-card">
                        <div style="display:flex; justify-content:space-between; align-items:flex-start; margin-bottom:10px;">
                            <div style="flex:1; padding-right:16px;">
                                <div style="font-family:'Space Grotesk',sans-serif; font-weight:600; font-size:0.88rem; color:#d4d4f0; margin-bottom:4px; line-height:1.4;">{article['title']}</div>
                                <div style="font-size:0.68rem; color:#5a5a82; margin-bottom:6px; font-family:'Fragment Mono',monospace;">{article['source']} · {pub_date}</div>
                                <div style="font-size:0.76rem; color:#6a6a92; font-family:'Fragment Mono',monospace; line-height:1.6;">{explanation}</div>
                            </div>
                            <div style="text-align:right; min-width:80px;">
                                <div style="font-family:'Bebas Neue',sans-serif; font-size:1.6rem; color:{bar_color}; letter-spacing:0.04em; line-height:1;">{sim:.0%}</div>
                                <span class="tag {tag_class}">{rel_label}</span>
                            </div>
                        </div>
                        <div style="background:#0d0d18; border-radius:1px; height:2px; margin-top:10px;">
                            <div style="background:{bar_color}; width:{bar_width}%; height:2px; border-radius:1px; opacity:0.7;"></div>
                        </div>
                        <div style="margin-top:8px;">{read_link}</div>
                    </div>
                    """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown('<div style="font-family:\'Fragment Mono\',monospace; font-size:0.62rem; color:#5a5a82; letter-spacing:0.2em; text-transform:uppercase; margin-bottom:12px;">Search Intelligence Log</div>', unsafe_allow_html=True)

        col_s1, col_s2, col_s3 = st.columns(3)
        with col_s1:
            st.markdown(f"""
            <div style="background:var(--surface); border:1px solid var(--border2); border-radius:4px; padding:14px 16px;">
                <div style="font-family:'Fragment Mono',monospace; font-size:0.6rem; color:#5a5a82; letter-spacing:0.15em; text-transform:uppercase; margin-bottom:6px;">Primary Query</div>
                <div style="font-family:'Space Grotesk',sans-serif; font-weight:700; color:#5b8fff; font-size:0.85rem; line-height:1.4;">{query_meta.get('primary_query','')}</div>
            </div>
            """, unsafe_allow_html=True)
        with col_s2:
            topics = ', '.join(query_meta.get('key_topics', [])[:4])
            st.markdown(f"""
            <div style="background:var(--surface); border:1px solid var(--border2); border-radius:4px; padding:14px 16px;">
                <div style="font-family:'Fragment Mono',monospace; font-size:0.6rem; color:#5a5a82; letter-spacing:0.15em; text-transform:uppercase; margin-bottom:6px;">Key Topics</div>
                <div style="font-family:'Space Grotesk',sans-serif; font-weight:700; color:#39d98a; font-size:0.82rem; line-height:1.4;">{topics or 'N/A'}</div>
            </div>
            """, unsafe_allow_html=True)
        with col_s3:
            st.markdown(f"""
            <div style="background:var(--surface); border:1px solid var(--border2); border-radius:4px; padding:14px 16px;">
                <div style="font-family:'Fragment Mono',monospace; font-size:0.6rem; color:#5a5a82; letter-spacing:0.15em; text-transform:uppercase; margin-bottom:6px;">Articles Analyzed</div>
                <div style="font-family:'Bebas Neue',sans-serif; font-size:1.8rem; color:#f5a623; letter-spacing:0.04em; line-height:1;">{len(tracker.articles)}</div>
            </div>
            """, unsafe_allow_html=True)

    elif not analyze_btn and not st.session_state.error_message:
        st.markdown("""
        <div style="background:var(--surface); border:1px solid var(--border2); border-radius:4px; padding:48px 32px; text-align:center; margin-top:20px; position:relative; overflow:hidden;">
            <div style="position:absolute; inset:0; background: radial-gradient(ellipse at center, rgba(91,143,255,0.04) 0%, transparent 70%); pointer-events:none;"></div>
            <div style="font-family:'Bebas Neue',sans-serif; font-size:3.5rem; color:rgba(91,143,255,0.12); letter-spacing:0.1em; line-height:1; margin-bottom:16px;">AWAITING INPUT</div>
            <div style="font-family:'Space Grotesk',sans-serif; font-weight:600; font-size:1rem; color:#5a5a82; margin-bottom:10px;">Ready to verify claims</div>
            <div style="font-size:0.78rem; color:#3a3a58; font-family:'Fragment Mono',monospace; line-height:2.1;">
                Add your free Groq API key + NewsAPI key in the sidebar<br>
                Enter any claim in any language<br>
                Get AI-powered semantic fact-checking with real news data
            </div>
        </div>
        """, unsafe_allow_html=True)

# ============================================
# TAB 2: VISUALIZATION
# ============================================
with tab_viz:
    render_visualization_tab(st.session_state.analysis_history)

# ============================================
# TAB 3: HISTORY & PATTERNS
# ============================================
with tab_history:
    render_history_tab(st.session_state.analysis_history)

# ---- FOOTER ----
st.markdown("---")
st.markdown("""
<div style="font-size:0.62rem; color:#1e1e35; font-family:'Fragment Mono',monospace; text-align:center; line-height:2.4; letter-spacing:0.06em;">
    Groq Llama 3.3 &nbsp;·&nbsp; NewsAPI.org &nbsp;·&nbsp; TruthLens v2.0
</div>
""", unsafe_allow_html=True)