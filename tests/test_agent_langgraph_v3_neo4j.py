"""Tests unitaires 13A sans Neo4j, Groq ni accès réseau."""
from __future__ import annotations

import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from agent_langgraph_v3_neo4j import (
    Q_LABELS,
    Q_NODES,
    Q_RELS,
    Q_TYPES,
    neo4j_metadata,
    read_neo4j_settings,
    route_question,
    validate_neo4j_trace,
)


class Record(dict):
    def data(self):
        return dict(self)


class FakeDriver:
    def __init__(self):
        self.calls = []
        self.payloads = {
            Q_NODES: [Record(total_nodes=5)],
            Q_RELS: [Record(total_relationships=3)],
            Q_LABELS: [Record(label="Condition"), Record(label="Article")],
            Q_TYPES: [Record(relationshipType="MENTIONS")],
        }

    def execute_query(self, query, **kwargs):
        self.calls.append((query, kwargs))
        return self.payloads[query], None, []


def tool(name, payload):
    return SimpleNamespace(type="tool", name=name, content=json.dumps(payload))


def final(text="anything"):
    return SimpleNamespace(type="ai", content=text, tool_calls=[])


class Neo4jToolTests(unittest.TestCase):
    def test_fixed_read_queries_and_database(self):
        fake = FakeDriver()
        info = neo4j_metadata(fake, "neo4j")
        self.assertEqual(info["total_nodes"], 5)
        self.assertEqual(info["total_relationships"], 3)
        self.assertEqual(info["labels_sample"], ["Condition", "Article"])
        self.assertEqual(len(fake.calls), 4)
        self.assertEqual([x[0] for x in fake.calls], [Q_NODES, Q_RELS, Q_LABELS, Q_TYPES])
        self.assertTrue(all(kwargs == {"database_": "neo4j", "routing_": "r"} for _, kwargs in fake.calls))
        self.assertFalse(any(" SET " in q.upper() or " DELETE " in q.upper() or " MERGE " in q.upper() for q, _ in fake.calls))

    def test_empty_graph_is_valid(self):
        fake = FakeDriver()
        fake.payloads[Q_NODES] = [Record(total_nodes=0)]
        fake.payloads[Q_RELS] = [Record(total_relationships=0)]
        fake.payloads[Q_LABELS] = []
        fake.payloads[Q_TYPES] = []
        self.assertEqual(neo4j_metadata(fake, "neo4j")["total_nodes"], 0)

    def test_bad_count_rejected(self):
        fake = FakeDriver()
        fake.payloads[Q_NODES] = [Record(total_nodes="five")]
        with self.assertRaisesRegex(ValueError, "invalide"):
            neo4j_metadata(fake, "neo4j")

    def test_route_graph_metadata(self):
        self.assertEqual(route_question("How many nodes and relationships are in Neo4j?"), "graph_metadata")
        self.assertEqual(route_question("Quels labels existent dans le graphe Neo4j ?"), "graph_metadata")

    def test_route_medquad_and_stats_unaffected(self):
        self.assertEqual(route_question("What are the symptoms of acromegaly?"), "search")
        self.assertEqual(route_question("How many documents and chunks are indexed?"), "statistics")

    def test_medical_current_preflight_has_priority(self):
        self.assertEqual(route_question("What do current clinical guidelines say in Neo4j?"), "blocked")

    def test_entitiy_graph_queries_not_claimed(self):
        self.assertEqual(route_question("Which diseases are connected to acromegaly in Neo4j?"), "graph_entities_not_ready")

    def test_read_credentials_never_printed(self):
        with patch.dict("os.environ", {"NEO4J_URI": "neo4j+s://example.databases.neo4j.io", "NEO4J_USERNAME": "reader", "NEO4J_PASSWORD": "secret-value", "NEO4J_DATABASE": "neo4j"}):
            uri, user, password, db = read_neo4j_settings()
            self.assertEqual((user, password, db), ("reader", "secret-value", "neo4j"))
            self.assertTrue(uri.startswith("neo4j+s://"))

    def test_missing_credentials(self):
        with patch.dict("os.environ", {"NEO4J_URI": "", "NEO4J_USERNAME": "", "NEO4J_USER": "", "NEO4J_PASSWORD": ""}):
            with self.assertRaises(EnvironmentError):
                read_neo4j_settings()

    def test_guard_accepts_valid_tool_data_not_llm_number(self):
        metadata = neo4j_metadata(FakeDriver(), "neo4j")
        check = validate_neo4j_trace({"messages": [tool("statistiques_graphe_neo4j", metadata), final("I invented 999 nodes")]})
        self.assertTrue(check["allowed"])
        self.assertEqual(check["metadata"]["total_nodes"], 5)

    def test_guard_rejects_wrong_tool(self):
        check = validate_neo4j_trace({"messages": [tool("statistiques_corpus", {"documents": 200}), final()]})
        self.assertEqual(check["reason"], "required_tool_missing:statistiques_graphe_neo4j")

    def test_guard_rejects_error_payload(self):
        check = validate_neo4j_trace({"messages": [tool("statistiques_graphe_neo4j", {"type": "neo4j_error", "reason": "ServiceUnavailable"}), final()]})
        self.assertFalse(check["allowed"])

    def test_guard_rejects_fake_count(self):
        metadata = neo4j_metadata(FakeDriver(), "neo4j")
        metadata["total_nodes"] = "five"
        check = validate_neo4j_trace({"messages": [tool("statistiques_graphe_neo4j", metadata), final()]})
        self.assertEqual(check["reason"], "invalid_graph_counts")


if __name__ == "__main__":
    unittest.main()
