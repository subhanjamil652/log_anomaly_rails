"""
Drain Log Parser - Implementation of the Drain algorithm for log template extraction.
Reference: He et al. (2016) - An Evaluation Study on Log Parsing and Its Use in Log Mining.

Falls back to drain3 library if available, otherwise uses built-in implementation.
"""

import re
import hashlib
import logging
from typing import Optional

logger = logging.getLogger(__name__)


# -------------------------------------------------
# Built-in Drain implementation (no external deps)
# -------------------------------------------------

class LogCluster:
    def __init__(self, template_tokens: list, cluster_id: int):
        self.template_tokens = template_tokens
        self.cluster_id = cluster_id
        self.size = 1

    def get_template(self) -> str:
        return " ".join(self.template_tokens)


class DrainNode:
    def __init__(self):
        self.children: dict = {}
        self.clusters: list = []


class DrainParser:
    """
    Drain log parser with O(n) complexity per log message.
    Extracts structured event templates from raw unstructured log messages.
    """

    # BGL log line format:
    # LABEL TIMESTAMP DATE TIME NODE TYPE TYPE_DETAIL COMPONENT MESSAGE
    BGL_PATTERN = re.compile(
        r'^(\S+)\s+'          # label (- = normal, else = alert type)
        r'(\d+)\s+'           # timestamp (epoch)
        r'(\S+)\s+'           # date (YYYY.MM.DD)
        r'(\S+)\s+'           # time (HH:MM:SS.mmm)
        r'(\S+)\s+'           # node_id
        r'(\S+)\s+'           # type (RAS/APP)
        r'(\S+)\s+'           # subtype
        r'(\S+)\s+'           # component
        r'(.+)$'              # message content
    )

    VARIABLE_PATTERNS = [
        re.compile(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b'),  # IP
        re.compile(r'\b[0-9a-fA-F]{8,}\b'),                       # hex
        re.compile(r'\b\d+\b'),                                     # numbers
        re.compile(r'(?<=[^A-Za-z])\d+(?=[^A-Za-z])|(?<=[A-Za-z])\d+'),
    ]

    def __init__(self, depth: int = 4, sim_threshold: float = 0.4,
                 max_children: int = 100, max_clusters: int = 1024):
        self.depth = max(depth - 2, 1)
        self.sim_threshold = sim_threshold
        self.max_children = max_children
        self.max_clusters = max_clusters
        self.root = DrainNode()
        self._id_counter = 0
        self._clusters: dict[int, LogCluster] = {}
        self._use_drain3 = False

        try:
            from drain3 import TemplateMiner
            from drain3.template_miner_config import TemplateMinerConfig
            config = TemplateMinerConfig()
            config.drain_depth = depth
            config.drain_sim_th = sim_threshold
            config.drain_max_children = max_children
            self._drain3_miner = TemplateMiner(config=config)
            self._use_drain3 = True
            logger.info("drain3 library loaded - using optimised implementation")
        except ImportError:
            logger.info("drain3 not available - using built-in Drain implementation")

    # -- Public API ---------------------------------

    def parse_bgl_line(self, line: str) -> dict:
        """Parse a single raw BGL log line into structured fields."""
        line = line.strip()
        m = self.BGL_PATTERN.match(line)
        if m:
            label, ts, date, time_, node, type_, subtype, component, content = m.groups()
            is_anomaly = 0 if label == "-" else 1
            template_info = self.parse(content)
            return {
                "label": label,
                "timestamp": int(ts),
                "date": date,
                "time": time_,
                "node": node,
                "type": type_,
                "component": component,
                "content": content,
                "is_anomaly": is_anomaly,
                "template": template_info["template"],
                "cluster_id": template_info["cluster_id"],
            }
        # fallback for lines that don't match full pattern
        template_info = self.parse(line)
        return {
            "label": "-",
            "timestamp": 0,
            "date": "",
            "time": "",
            "node": "",
            "type": "UNKNOWN",
            "component": "UNKNOWN",
            "content": line,
            "is_anomaly": 0,
            "template": template_info["template"],
            "cluster_id": template_info["cluster_id"],
        }

    def parse(self, log_message: str) -> dict:
        """Extract template from a single log message string."""
        if self._use_drain3:
            result = self._drain3_miner.add_log_message(log_message)
            return {
                "template": result["template_mined"] if result else log_message,
                "params": [],
                "cluster_id": result["cluster_id"] if result else 0,
            }
        tokens = self._tokenise(log_message)
        cluster = self._tree_search(tokens)
        if cluster is None:
            cluster = self._create_cluster(tokens)
        else:
            cluster = self._update_cluster(cluster, tokens)
        params = self._extract_params(cluster.template_tokens, tokens)
        return {
            "template": cluster.get_template(),
            "params": params,
            "cluster_id": cluster.cluster_id,
        }

    def parse_batch(self, log_messages: list) -> list:
        """Parse a list of log message strings."""
        return [self.parse(msg) for msg in log_messages]

    def get_templates(self) -> dict:
        """Return all discovered templates keyed by cluster_id."""
        if self._use_drain3:
            return {c.cluster_id: c.get_template()
                    for c in self._drain3_miner.drain.clusters}
        return {cid: c.get_template() for cid, c in self._clusters.items()}

    # -- Internal Drain algorithm -------------------

    def _tokenise(self, message: str) -> list:
        tokens = message.split()
        # replace obvious variables with <*>
        cleaned = []
        for tok in tokens:
            if re.fullmatch(r'\d+', tok):
                cleaned.append("<*>")
            elif re.fullmatch(r'[0-9a-fA-F]{8,}', tok):
                cleaned.append("<*>")
            elif re.fullmatch(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', tok):
                cleaned.append("<*>")
            else:
                cleaned.append(tok)
        return cleaned

    def _tree_search(self, tokens: list) -> Optional[LogCluster]:
        n = len(tokens)
        if n not in self.root.children:
            return None
        len_node = self.root.children[n]

        # first-token routing
        first_tok = tokens[0] if tokens else ""
        key = first_tok if (not re.fullmatch(r'\d+|\S*\d\S*', first_tok)) else "<*>"
        if key not in len_node.children:
            if "<*>" not in len_node.children:
                return None
            key = "<*>"
        next_node = len_node.children[key]

        return self._cluster_search(next_node.clusters, tokens)

    def _cluster_search(self, clusters: list, tokens: list) -> Optional[LogCluster]:
        best_sim, best_cluster = -1.0, None
        for cluster in clusters:
            sim = self._seq_sim(cluster.template_tokens, tokens)
            if sim > best_sim:
                best_sim, best_cluster = sim, cluster
        if best_sim >= self.sim_threshold:
            return best_cluster
        return None

    def _seq_sim(self, tmpl: list, tokens: list) -> float:
        if len(tmpl) != len(tokens):
            return 0.0
        matches = sum(1 for t, s in zip(tmpl, tokens)
                      if t == s or t == "<*>")
        return matches / len(tmpl) if tmpl else 0.0

    def _create_cluster(self, tokens: list) -> LogCluster:
        self._id_counter += 1
        cluster = LogCluster(list(tokens), self._id_counter)
        self._clusters[self._id_counter] = cluster
        self._add_to_tree(cluster, tokens)
        return cluster

    def _add_to_tree(self, cluster: LogCluster, tokens: list):
        n = len(tokens)
        if n not in self.root.children:
            self.root.children[n] = DrainNode()
        len_node = self.root.children[n]

        first_tok = tokens[0] if tokens else "<*>"
        key = first_tok if not re.fullmatch(r'\d+', first_tok) else "<*>"
        if key not in len_node.children:
            if len(len_node.children) >= self.max_children:
                key = "<*>"
            len_node.children[key] = DrainNode()
        len_node.children[key].clusters.append(cluster)

    def _update_cluster(self, cluster: LogCluster, tokens: list) -> LogCluster:
        new_tmpl = []
        for t, s in zip(cluster.template_tokens, tokens):
            new_tmpl.append(t if t == s else "<*>")
        cluster.template_tokens = new_tmpl
        cluster.size += 1
        return cluster

    def _extract_params(self, template: list, tokens: list) -> list:
        return [tok for tmpl_tok, tok in zip(template, tokens)
                if tmpl_tok == "<*>"]
