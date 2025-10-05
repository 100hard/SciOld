"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import axios from "axios";
import {
  AlertCircle,
  ArrowUpRight,
  Loader2,
  RefreshCw,
  SlidersHorizontal,
  Trash2,
} from "lucide-react";

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

const DEFAULT_DIMENSIONS = {
  width: 960,
  height: 640,
};

const GRAPH_LIMIT = 150;
const NEIGHBORHOOD_LIMIT = 75;

type NodeType =
  | "method"
  | "dataset"
  | "metric"
  | "task"
  | "concept"
  | "material"
  | "organism"
  | "finding"
  | "process";
type RelationType =
  | "proposes"
  | "evaluates_on"
  | "reports"
  | "compares"
  | "uses"
  | "causes"
  | "part_of"
  | "is_a"
  | "outperforms"
  | "assumes";

type GraphNodeLink = {
  id: string;
  label: string;
  type: NodeType;
  relation: RelationType;
  weight: number;
};

type GraphEvidenceItem = {
  paper_id: string;
  paper_title?: string | null;
  snippet?: string | null;
  confidence: number;
  relation: RelationType;
};

type GraphNodeData = {
  id: string;
  type: NodeType;
  label: string;
  entity_id: string;
  paper_count: number;
  aliases: string[];
  description?: string | null;
  top_links: GraphNodeLink[];
  evidence: GraphEvidenceItem[];
  metadata?: Record<string, unknown> | null;
};

type GraphNode = {
  data: GraphNodeData;
};

type GraphEdgeData = {
  id: string;
  source: string;
  target: string;
  type: RelationType;
  weight: number;
  paper_count: number;
  average_confidence: number;
  metadata?: Record<string, unknown> | null;
};

type GraphEdge = {
  data: GraphEdgeData;
};

type NodePapersByYear = {
  year: number;
  paper_count: number;
};

type NodeOutcomeMetadata = {
  paper_id: string;
  paper_title?: string | null;
  paper_year?: number | null;
  value_numeric?: number | null;
  value_text?: string | null;
  metric?: string | null;
  metric_unit?: string | null;
  dataset?: string | null;
  task?: string | null;
  confidence?: number | null;
  is_sota?: boolean | null;
  verified?: boolean | null;
};

type EdgeInsight = {
  summary: string;
  paper_id: string;
  paper_title?: string | null;
  paper_year?: number | null;
  confidence?: number | null;
  claim_text?: string | null;
  metric?: string | null;
  dataset?: string | null;
  task?: string | null;
  value_text?: string | null;
  value_numeric?: number | null;
  relation?: RelationType;
};

type EdgeInsightWithContext = EdgeInsight & {
  relation: RelationType;
  connectedNodeId: string;
};

type GraphMeta = {
  limit: number;
  node_count: number;
  edge_count: number;
  concept_count?: number | null;
  paper_count?: number | null;
  has_more?: boolean | null;
  center_id?: string | null;
  center_type?: NodeType | null;
  filters?: {
    types?: string[];
    relations?: string[];
    min_conf?: number;
  } | null;
};

type GraphResponse = {
  nodes: GraphNode[];
  edges: GraphEdge[];
  meta: GraphMeta;
};

type GraphState = {
  nodes: Record<string, GraphNodeData>;
  edges: Record<string, GraphEdgeData>;
};

type LayoutDimensions = {
  width: number;
  height: number;
};

type LayoutPositions = Record<string, { x: number; y: number }>;

const INITIAL_GRAPH_STATE: GraphState = {
  nodes: {},
  edges: {},
};

const GRAPH_CLEARED_STORAGE_KEY = "scinets.graphCleared";

const ALL_TYPES: NodeType[] = [
  "method",
  "dataset",
  "metric",
  "task",
  "concept",
  "material",
  "organism",
  "finding",
  "process",
];
const ALL_RELATIONS: RelationType[] = [
  "proposes",
  "evaluates_on",
  "reports",
  "compares",
  "uses",
  "causes",
  "part_of",
  "is_a",
  "outperforms",
  "assumes",
];

const NODE_COLORS: Record<NodeType, string> = {
  method: "#0ea5e9",
  dataset: "#22c55e",
  metric: "#8b5cf6",
  task: "#f97316",
  concept: "#14b8a6",
  material: "#b45309",
  organism: "#10b981",
  finding: "#ef4444",
  process: "#6366f1",
};

const EDGE_COLORS: Record<RelationType, string> = {
  proposes: "#f97316",
  evaluates_on: "#22c55e",
  reports: "#8b5cf6",
  compares: "#64748b",
  uses: "#0ea5e9",
  causes: "#ef4444",
  part_of: "#d946ef",
  is_a: "#6366f1",
  outperforms: "#10b981",
  assumes: "#facc15",
};

const clamp = (value: number, min: number, max: number) => Math.min(Math.max(value, min), max);

const toNumber = (value: unknown): number | undefined => {
  if (typeof value === "number" && Number.isFinite(value)) {
    return value;
  }
  if (typeof value === "string") {
    const parsed = Number(value);
    if (!Number.isNaN(parsed)) {
      return parsed;
    }
  }
  return undefined;
};

const normalizePapersByYear = (value: unknown): NodePapersByYear[] => {
  if (!Array.isArray(value)) {
    return [];
  }
  const entries: NodePapersByYear[] = [];
  for (const entry of value) {
    if (!entry || typeof entry !== "object") {
      continue;
    }
    const record = entry as Record<string, unknown>;
    const year = toNumber(record.year);
    const paperCount = toNumber(record.paper_count);
    if (typeof year === "number" && typeof paperCount === "number") {
      entries.push({ year, paper_count: paperCount });
    }
  }
  return entries.sort((a, b) => b.year - a.year);
};

const normalizeOutcomeMetadata = (value: unknown): NodeOutcomeMetadata | undefined => {
  if (!value || typeof value !== "object") {
    return undefined;
  }
  const record = value as Record<string, unknown>;
  const paperId = typeof record.paper_id === "string" ? record.paper_id : undefined;
  if (!paperId) {
    return undefined;
  }
  const outcome: NodeOutcomeMetadata = { paper_id: paperId };
  if (typeof record.paper_title === "string") {
    outcome.paper_title = record.paper_title;
  }
  const year = toNumber(record.paper_year);
  if (typeof year === "number") {
    outcome.paper_year = year;
  }
  const valueNumeric = toNumber(record.value_numeric);
  if (typeof valueNumeric === "number") {
    outcome.value_numeric = valueNumeric;
  }
  if (typeof record.value_text === "string" && record.value_text.trim()) {
    outcome.value_text = record.value_text.trim();
  }
  if (typeof record.metric === "string") {
    outcome.metric = record.metric;
  }
  if (typeof record.metric_unit === "string") {
    outcome.metric_unit = record.metric_unit;
  }
  if (typeof record.dataset === "string") {
    outcome.dataset = record.dataset;
  }
  if (typeof record.task === "string") {
    outcome.task = record.task;
  }
  const confidence = toNumber(record.confidence);
  if (typeof confidence === "number") {
    outcome.confidence = confidence;
  }
  if (typeof record.is_sota === "boolean") {
    outcome.is_sota = record.is_sota;
  }
  if (typeof record.verified === "boolean") {
    outcome.verified = record.verified;
  }
  return outcome;
};

const normalizeEdgeInsights = (value: unknown): EdgeInsight[] => {
  if (!Array.isArray(value)) {
    return [];
  }
  const insights: EdgeInsight[] = [];
  for (const entry of value) {
    if (!entry || typeof entry !== "object") {
      continue;
    }
    const record = entry as Record<string, unknown>;
    const summary = typeof record.summary === "string" ? record.summary.trim() : "";
    const paperId = typeof record.paper_id === "string" ? record.paper_id : undefined;
    if (!summary || !paperId) {
      continue;
    }
    const insight: EdgeInsight = { summary, paper_id: paperId };
    if (typeof record.paper_title === "string") {
      insight.paper_title = record.paper_title;
    }
    const year = toNumber(record.paper_year);
    if (typeof year === "number") {
      insight.paper_year = year;
    }
    const confidence = toNumber(record.confidence);
    if (typeof confidence === "number") {
      insight.confidence = confidence;
    }
    if (typeof record.claim_text === "string") {
      insight.claim_text = record.claim_text;
    }
    if (typeof record.metric === "string") {
      insight.metric = record.metric;
    }
    if (typeof record.dataset === "string") {
      insight.dataset = record.dataset;
    }
    if (typeof record.task === "string") {
      insight.task = record.task;
    }
    if (typeof record.value_text === "string") {
      insight.value_text = record.value_text;
    }
    const valueNumeric = toNumber(record.value_numeric);
    if (typeof valueNumeric === "number") {
      insight.value_numeric = valueNumeric;
    }
    if (typeof record.relation === "string" && (ALL_RELATIONS as string[]).includes(record.relation)) {
      insight.relation = record.relation as RelationType;
    }
    insights.push(insight);
  }
  return insights;
};

const formatOutcomeSummary = (outcome: NodeOutcomeMetadata): string => {
  const parts: string[] = [];
  if (outcome.metric) {
    parts.push(outcome.metric);
  }
  const valuePieces: string[] = [];
  if (typeof outcome.value_text === "string") {
    valuePieces.push(outcome.value_text);
  } else if (typeof outcome.value_numeric === "number") {
    valuePieces.push(outcome.value_numeric.toFixed(2));
  }
  if (outcome.metric_unit && valuePieces.length > 0) {
    valuePieces[valuePieces.length - 1] = `${valuePieces[valuePieces.length - 1]}${outcome.metric_unit}`;
  }
  if (valuePieces.length > 0) {
    parts.push(valuePieces.join(" "));
  }
  if (outcome.dataset) {
    parts.push(`on ${outcome.dataset}`);
  } else if (outcome.task) {
    parts.push(`for ${outcome.task}`);
  }
  return parts.join(" ").trim();
};

const formatOutcomeDetails = (outcome: NodeOutcomeMetadata): string => {
  const details: string[] = [];
  if (typeof outcome.paper_year === "number") {
    details.push(String(outcome.paper_year));
  }
  if (typeof outcome.confidence === "number") {
    details.push(`${Math.round(outcome.confidence * 100)}% confidence`);
  }
  if (outcome.is_sota) {
    details.push("SOTA");
  }
  if (outcome.verified) {
    details.push("Tier-3 verified");
  }
  return details.join(" • ");
};

const isPlaceholderMetadata = (metadata?: Record<string, unknown> | null): boolean => {
  if (!metadata || typeof metadata !== "object") {
    return false;
  }
  return (metadata as Record<string, unknown>).placeholder === true;
};

const formatMetadataValue = (value: unknown): string => {
  if (value === null || value === undefined) {
    return "—";
  }
  if (typeof value === "string" || typeof value === "number" || typeof value === "bigint") {
    return String(value);
  }
  if (typeof value === "boolean") {
    return value ? "Yes" : "No";
  }
  try {
    return JSON.stringify(value);
  } catch (error) {
    return String(value);
  }
};

const formatTypeLabel = (type: NodeType) => {
  switch (type) {
    case "method":
      return "Method";
    case "dataset":
      return "Dataset";
    case "metric":
      return "Metric";
    case "task":
      return "Task";
    case "concept":
      return "Concept";
    case "material":
      return "Material";
    case "organism":
      return "Organism";
    case "finding":
      return "Finding";
    case "process":
      return "Process";
    default:
      return type;
  }
};

const formatRelationLabel = (relation: RelationType) => {
  switch (relation) {
    case "proposes":
      return "Proposes";
    case "evaluates_on":
      return "Evaluates on";
    case "reports":
      return "Reports";
    case "compares":
      return "Compares";
    case "uses":
      return "Uses";
    case "causes":
      return "Causes";
    case "part_of":
      return "Part of";
    case "is_a":
      return "Is a";
    case "outperforms":
      return "Outperforms";
    case "assumes":
      return "Assumes";
    default:
      return relation;
  }
};

const MAX_FORCE_ITERATIONS = 320;
const MIN_FORCE_ITERATIONS = 120;
const REPULSION_BASE = 1600;
const SPRING_LENGTH = 160;
const SPRING_STRENGTH = 0.015;
const DAMPING = 0.82;
const CENTER_STRENGTH = 0.0065;

const computeLayout = (
  nodes: GraphNodeData[],
  edges: GraphEdgeData[],
  dimensions: LayoutDimensions,
  previousPositions: LayoutPositions,
): LayoutPositions => {
  if (!nodes.length) {
    return {};
  }

  const width = Math.max(dimensions.width, DEFAULT_DIMENSIONS.width);
  const height = Math.max(dimensions.height, DEFAULT_DIMENSIONS.height);
  const centerX = width / 2;
  const centerY = height / 2;

  const positions: LayoutPositions = {};
  const velocities: Record<string, { x: number; y: number }> = {};

  nodes.forEach((node, index) => {
    const previous = previousPositions[node.id];
    const angle = (2 * Math.PI * index) / nodes.length;
    positions[node.id] = previous
      ? { ...previous }
      : {
          x: centerX + Math.cos(angle) * (width * 0.25),
          y: centerY + Math.sin(angle) * (height * 0.25),
        };
    velocities[node.id] = { x: 0, y: 0 };
  });

  const nodeCount = nodes.length;
  const iterations = Math.min(
    MAX_FORCE_ITERATIONS,
    Math.max(MIN_FORCE_ITERATIONS, nodeCount * 4),
  );
  const repulsionStrength = REPULSION_BASE + nodeCount * 35;

  const forcesTemplate: Record<string, { x: number; y: number }> = {};
  nodes.forEach((node) => {
    forcesTemplate[node.id] = { x: 0, y: 0 };
  });

  for (let iteration = 0; iteration < iterations; iteration += 1) {
    const forces: Record<string, { x: number; y: number }> = {};
    nodes.forEach((node) => {
      forces[node.id] = { x: 0, y: 0 };
    });

    for (let i = 0; i < nodeCount; i += 1) {
      const source = nodes[i];
      const sourcePos = positions[source.id];
      for (let j = i + 1; j < nodeCount; j += 1) {
        const target = nodes[j];
        const targetPos = positions[target.id];
        const dx = sourcePos.x - targetPos.x;
        const dy = sourcePos.y - targetPos.y;
        const distanceSq = dx * dx + dy * dy + 0.01;
        const distance = Math.sqrt(distanceSq);
        const force = repulsionStrength / distanceSq;
        const fx = (dx / distance) * force;
        const fy = (dy / distance) * force;
        forces[source.id].x += fx;
        forces[source.id].y += fy;
        forces[target.id].x -= fx;
        forces[target.id].y -= fy;
      }
    }

    edges.forEach((edge) => {
      const source = positions[edge.source];
      const target = positions[edge.target];
      if (!source || !target) {
        return;
      }
      const dx = source.x - target.x;
      const dy = source.y - target.y;
      const distance = Math.sqrt(dx * dx + dy * dy) || 0.01;
      const force = SPRING_STRENGTH * (distance - SPRING_LENGTH);
      const fx = (dx / distance) * force;
      const fy = (dy / distance) * force;
      forces[edge.source].x -= fx;
      forces[edge.source].y -= fy;
      forces[edge.target].x += fx;
      forces[edge.target].y += fy;
    });

    nodes.forEach((node) => {
      const pos = positions[node.id];
      const centerFx = (centerX - pos.x) * CENTER_STRENGTH;
      const centerFy = (centerY - pos.y) * CENTER_STRENGTH;
      forces[node.id].x += centerFx;
      forces[node.id].y += centerFy;
    });

    nodes.forEach((node) => {
      const velocity = velocities[node.id];
      const force = forces[node.id];
      velocity.x = (velocity.x + force.x) * DAMPING;
      velocity.y = (velocity.y + force.y) * DAMPING;
      const pos = positions[node.id];
      pos.x += velocity.x;
      pos.y += velocity.y;
    });
  }

  const margin = 60;
  const clampedPositions: LayoutPositions = {};
  nodes.forEach((node) => {
    const pos = positions[node.id];
    clampedPositions[node.id] = {
      x: clamp(pos.x, margin, width - margin),
      y: clamp(pos.y, margin, height - margin),
    };
  });

  return clampedPositions;
};

const getErrorMessage = (error: unknown): string => {
  if (axios.isAxiosError(error)) {
    const detail = error.response?.data as unknown;
    if (detail && typeof detail === "object" && "detail" in detail && typeof (detail as any).detail === "string") {
      return (detail as any).detail;
    }
    if (typeof error.message === "string" && error.message.trim().length > 0) {
      return error.message;
    }
  }
  if (error instanceof Error) {
    return error.message;
  }
  return "Unexpected error while fetching graph data.";
};

const GraphExplorer = () => {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const initialGraphCleared = (
    typeof window !== "undefined" && window.localStorage.getItem(GRAPH_CLEARED_STORAGE_KEY) === "true"
  );
  const [dimensions, setDimensions] = useState<LayoutDimensions>(DEFAULT_DIMENSIONS);
  const [graph, setGraph] = useState<GraphState>(INITIAL_GRAPH_STATE);
  const [meta, setMeta] = useState<GraphMeta | null>(null);
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null);
  const [expandedNodes, setExpandedNodes] = useState<Record<string, boolean>>({});
  const [error, setError] = useState<string | null>(null);
  const [isInitialLoading, setIsInitialLoading] = useState<boolean>(!initialGraphCleared);
  const [isClearing, setIsClearing] = useState<boolean>(false);
  const [expandingNodeId, setExpandingNodeId] = useState<string | null>(null);
  const [selectedTypes, setSelectedTypes] = useState<NodeType[]>(ALL_TYPES);
  const [isGraphCleared, setIsGraphCleared] = useState<boolean>(initialGraphCleared);
  const [selectedRelations, setSelectedRelations] = useState<RelationType[]>(ALL_RELATIONS);
  const [minConfidence, setMinConfidence] = useState<number>(0.6);
  const [showEvidence, setShowEvidence] = useState<boolean>(false);
  const previousPositionsRef = useRef<LayoutPositions>({});

  useEffect(() => {
    if (typeof window === "undefined") {
      return;
    }
    if (isGraphCleared) {
      window.localStorage.setItem(GRAPH_CLEARED_STORAGE_KEY, "true");
    } else {
      window.localStorage.removeItem(GRAPH_CLEARED_STORAGE_KEY);
    }
  }, [isGraphCleared]);

  useEffect(() => {
    const element = containerRef.current;
    if (!element) {
      return;
    }

    const observer = new ResizeObserver((entries) => {
      const entry = entries[0];
      if (!entry) {
        return;
      }
      const nextWidth = Math.max(entry.contentRect.width, DEFAULT_DIMENSIONS.width);
      const nextHeight = Math.max(entry.contentRect.height, DEFAULT_DIMENSIONS.height);
      setDimensions((current) => {
        if (Math.abs(current.width - nextWidth) < 1 && Math.abs(current.height - nextHeight) < 1) {
          return current;
        }
        return { width: nextWidth, height: nextHeight };
      });
    });

    observer.observe(element);

    return () => observer.disconnect();
  }, []);

  const mergeGraphData = useCallback((payload: GraphResponse, options?: { reset?: boolean; allowSelect?: boolean }) => {
    setGraph((previous) => {
      const nextNodes = options?.reset ? {} : { ...previous.nodes };
      payload.nodes.forEach((node) => {
        nextNodes[node.data.id] = {
          ...node.data,
          aliases: node.data.aliases ?? [],
          top_links: node.data.top_links ?? [],
          evidence: node.data.evidence ?? [],
          metadata: node.data.metadata ?? undefined,
        };
      });

      const nextEdges = options?.reset ? {} : { ...previous.edges };
      payload.edges.forEach((edge) => {
        nextEdges[edge.data.id] = {
          ...edge.data,
          metadata: edge.data.metadata ?? undefined,
        };
      });

      return {
        nodes: nextNodes,
        edges: nextEdges,
      };
    });

    setMeta(payload.meta);
    setError(null);

    if (options?.allowSelect) {
      setSelectedNodeId((current) => {
        if (current && payload.nodes.some((node) => node.data.id === current)) {
          return current;
        }
        return payload.meta.center_id ?? payload.nodes[0]?.data.id ?? current;
      });
    }
  }, []);

  const buildFilterParams = useCallback(() => {
    return {
      types: selectedTypes.join(","),
      relations: selectedRelations.join(","),
      min_conf: Number(minConfidence.toFixed(2)),
    };
  }, [minConfidence, selectedRelations, selectedTypes]);

  const loadOverview = useCallback(async () => {
    setIsInitialLoading(true);
    setError(null);
    setExpandedNodes({});
    try {
      const params = { limit: GRAPH_LIMIT, ...buildFilterParams() };
      const response = await axios.get<GraphResponse>(`${API_BASE_URL}/api/graph/overview`, {
        params,
      });
      mergeGraphData(response.data, { reset: true, allowSelect: true });
    } catch (err) {
      setError(getErrorMessage(err));
    } finally {
      setIsInitialLoading(false);
      setExpandingNodeId(null);
    }
  }, [buildFilterParams, mergeGraphData]);

  useEffect(() => {
    if (isGraphCleared) {
      return;
    }
    void loadOverview();
  }, [loadOverview, isGraphCleared]);

  useEffect(() => {
    if (selectedNodeId && !graph.nodes[selectedNodeId]) {
      const fallback = Object.keys(graph.nodes)[0] ?? null;
      setSelectedNodeId(fallback);
    }
  }, [graph.nodes, selectedNodeId]);

  useEffect(() => {
    setShowEvidence(false);
  }, [selectedNodeId]);

  const expandNode = useCallback(
    async (nodeId: string) => {
      const node = graph.nodes[nodeId];
      if (!node) {
        return;
      }
      if (expandedNodes[nodeId]) {
        return;
      }

      const targetId = node.entity_id;
      if (!targetId) {
        return;
      }

      if (isPlaceholderMetadata(node.metadata)) {
        return;
      }

      setExpandingNodeId(nodeId);
      try {
        const params = { limit: NEIGHBORHOOD_LIMIT, ...buildFilterParams() };
        const response = await axios.get<GraphResponse>(`${API_BASE_URL}/api/graph/neighborhood/${targetId}`, {
          params,
        });
        mergeGraphData(response.data, { allowSelect: false });
        setExpandedNodes((previous) => ({
          ...previous,
          [nodeId]: true,
        }));
      } catch (err) {
        setError(getErrorMessage(err));
      } finally {
        setExpandingNodeId((current) => (current === nodeId ? null : current));
      }
    },
    [buildFilterParams, expandedNodes, graph.nodes, mergeGraphData]
  );

  const handleNodeSelect = useCallback(
    (nodeId: string) => {
      setSelectedNodeId(nodeId);
      void expandNode(nodeId);
    },
    [expandNode]
  );

  const nodes = useMemo(() => Object.values(graph.nodes), [graph.nodes]);
  const edges = useMemo(() => Object.values(graph.edges), [graph.edges]);

  const positions = useMemo(
    () => computeLayout(nodes, edges, dimensions, previousPositionsRef.current),
    [nodes, edges, dimensions]
  );

  useEffect(() => {
    previousPositionsRef.current = positions;
  }, [positions]);

  const nodesWithPositions = useMemo(
    () =>
      nodes.map((node) => ({
        ...node,
        x: positions[node.id]?.x ?? dimensions.width / 2,
        y: positions[node.id]?.y ?? dimensions.height / 2,
      })),
    [nodes, positions, dimensions]
  );

  const selectedNode = selectedNodeId ? graph.nodes[selectedNodeId] : undefined;
  const selectedNodeIsPlaceholder = selectedNode ? isPlaceholderMetadata(selectedNode.metadata ?? undefined) : false;

  const { papersByYear, bestOutcome, worstOutcome, remainingMetadata } = useMemo(() => {
    if (!selectedNode?.metadata) {
      return {
        papersByYear: [] as NodePapersByYear[],
        bestOutcome: undefined as NodeOutcomeMetadata | undefined,
        worstOutcome: undefined as NodeOutcomeMetadata | undefined,
        remainingMetadata: {} as Record<string, unknown>,
      };
    }
    const raw = selectedNode.metadata as Record<string, unknown>;
    const { papers_by_year: rawPapersByYear, best_outcome: rawBestOutcome, worst_outcome: rawWorstOutcome, ...rest } = raw;
    return {
      papersByYear: normalizePapersByYear(rawPapersByYear),
      bestOutcome: normalizeOutcomeMetadata(rawBestOutcome),
      worstOutcome: normalizeOutcomeMetadata(rawWorstOutcome),
      remainingMetadata: rest,
    };
  }, [selectedNode?.metadata]);

  const nodeEdgeInsights = useMemo(() => {
    if (!selectedNode) {
      return [] as EdgeInsightWithContext[];
    }
    const insights: EdgeInsightWithContext[] = [];
    edges.forEach((edge) => {
      if (edge.source !== selectedNode.id && edge.target !== selectedNode.id) {
        return;
      }
      const connectedNodeId = edge.source === selectedNode.id ? edge.target : edge.source;
      const insightSource =
        edge.metadata && typeof edge.metadata === "object" && "insights" in edge.metadata
          ? (edge.metadata as Record<string, unknown>).insights
          : undefined;
      const rawInsights = normalizeEdgeInsights(insightSource);
      rawInsights.forEach((insight) => {
        insights.push({ ...insight, relation: edge.type, connectedNodeId });
      });
    });
    return insights;
  }, [edges, selectedNode]);

  const remainingMetadataEntries = useMemo(
    () => Object.entries(remainingMetadata ?? {}),
    [remainingMetadata]
  );

  const neighborSet = useMemo(() => {
    if (!selectedNodeId) {
      return new Set<string>();
    }
    const set = new Set<string>();
    edges.forEach((edge) => {
      if (edge.source === selectedNodeId) {
        set.add(edge.target);
      } else if (edge.target === selectedNodeId) {
        set.add(edge.source);
      }
    });
    return set;
  }, [edges, selectedNodeId]);

  const stats = useMemo(() => {
    const typeStats = ALL_TYPES.map((type) => ({
      label: `${formatTypeLabel(type)} nodes`,
      value: nodes.filter((node) => node.type === type).length,
    })).filter((item) => item.value > 0);
    return [
      { label: "Total nodes", value: nodes.length },
      { label: "Edges", value: edges.length },
      ...typeStats,
    ];
  }, [edges.length, nodes]);

  const graphSummary = useMemo(() => {
    if (!meta) {
      return null;
    }
    const pieces: string[] = [];
    if (meta.paper_count) {
      pieces.push(`${meta.paper_count} unique papers represented`);
    }
    if (meta.filters) {
      const types = meta.filters.types?.join(", ");
      const relations = meta.filters.relations?.join(", ");
      if (types) {
        pieces.push(`Types: ${types}`);
      }
      if (relations) {
        pieces.push(`Relations: ${relations}`);
      }
      if (typeof meta.filters.min_conf === "number") {
        pieces.push(`Min confidence: ${(meta.filters.min_conf * 100).toFixed(0)}%`);
      }
    }
    if (meta.has_more) {
      pieces.push("Additional results available — expand nodes to load more");
    }
    return pieces.join(" • ");
  }, [meta]);

  const handleRefresh = () => {
    setIsGraphCleared(false);
    void loadOverview();
  };

  const handleClearGraph = async () => {
    setIsClearing(true);
    setError(null);
    try {
      await axios.post(`${API_BASE_URL}/api/graph/clear`);
      setIsGraphCleared(true);
      previousPositionsRef.current = {};
      setGraph({ nodes: {}, edges: {} });
      setMeta(null);
      setSelectedNodeId(null);
      setExpandedNodes({});
      setShowEvidence(false);
      setExpandingNodeId(null);
      setIsInitialLoading(false);
    } catch (err) {
      setError(getErrorMessage(err));
    } finally {
      setIsClearing(false);
    }
  };

  const handleExpandSelected = () => {
    if (selectedNodeId) {
      void expandNode(selectedNodeId);
    }
  };

  const toggleType = (type: NodeType) => {
    setSelectedTypes((current) => {
      if (current.includes(type)) {
        if (current.length === 1) {
          return current;
        }
        return current.filter((item) => item !== type);
      }
      const withType = [...current, type];
      return ALL_TYPES.filter((item) => withType.includes(item));
    });
  };

  const toggleRelation = (relation: RelationType) => {
    setSelectedRelations((current) => {
      if (current.includes(relation)) {
        if (current.length === 1) {
          return current;
        }
        return current.filter((item) => item !== relation);
      }
      const withRelation = [...current, relation];
      return ALL_RELATIONS.filter((item) => withRelation.includes(item));
    });
  };

  const handleConfidenceChange = (value: number) => {
    setMinConfidence(clamp(Number(value), 0.5, 1));
  };

  const hasGraphData = nodesWithPositions.length > 0;

  return (
    <div className="grid gap-6 lg:grid-cols-[minmax(0,1fr)_360px]">
      <div className="space-y-5">
        <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-3">
          {stats.map((item) => (
            <div key={item.label} className="rounded-lg border bg-card p-4 shadow-sm transition hover:shadow-md">
              <p className="text-xs font-medium uppercase tracking-wide text-muted-foreground">{item.label}</p>
              <p className="mt-3 text-2xl font-semibold text-foreground">{item.value.toLocaleString()}</p>
            </div>
          ))}
        </div>

        <div className="rounded-lg border bg-card p-4 shadow-sm">
          <div className="flex flex-wrap items-center justify-between gap-3">
            <div className="flex items-center gap-2 text-sm font-medium text-foreground">
              <SlidersHorizontal className="h-4 w-4" />
              Filters
            </div>
            <div className="flex flex-wrap items-center gap-3 text-xs text-muted-foreground">
              <div className="flex items-center gap-2">
                {ALL_TYPES.map((type) => {
                  const isActive = selectedTypes.includes(type);
                  return (
                    <button
                      key={type}
                      type="button"
                      onClick={() => toggleType(type)}
                      className={`inline-flex items-center rounded-md border px-2.5 py-1 text-xs font-medium transition ${
                        isActive
                          ? "border-primary bg-primary/10 text-primary"
                          : "border-border bg-background text-muted-foreground hover:bg-muted"
                      }`}
                    >
                      {formatTypeLabel(type)}
                    </button>
                  );
                })}
              </div>
              <div className="flex items-center gap-2">
                <span className="uppercase tracking-wide">Confidence</span>
                <input
                  type="range"
                  min={0.5}
                  max={1}
                  step={0.05}
                  value={minConfidence}
                  onChange={(event) => handleConfidenceChange(Number(event.target.value))}
                  className="h-1 w-24 cursor-pointer"
                />
                <span className="font-semibold text-foreground">{(minConfidence * 100).toFixed(0)}%</span>
              </div>
              <div className="flex items-center gap-2">
                {ALL_RELATIONS.map((relation) => {
                  const isActive = selectedRelations.includes(relation);
                  return (
                    <button
                      key={relation}
                      type="button"
                      onClick={() => toggleRelation(relation)}
                      className={`inline-flex items-center rounded-md border px-2.5 py-1 text-xs font-medium transition ${
                        isActive
                          ? "border-secondary bg-secondary/10 text-secondary-foreground"
                          : "border-border bg-background text-muted-foreground hover:bg-muted"
                      }`}
                    >
                      {formatRelationLabel(relation)}
                    </button>
                  );
                })}
              </div>
            </div>
          </div>
        </div>

        {error ? (
          <div className="flex items-center gap-3 rounded-md border border-rose-200 bg-rose-50 p-4 text-sm text-rose-700">
            <AlertCircle className="h-4 w-4" />
            <span>{error}</span>
          </div>
        ) : null}

        <div className="overflow-hidden rounded-lg border bg-card shadow-sm">
          <div className="flex flex-wrap items-center justify-between gap-3 border-b px-4 py-3">
            <div>
              <h2 className="text-lg font-semibold text-foreground">Knowledge graph</h2>
              <p className="text-xs text-muted-foreground">Explore typed research entities and their structured relationships.</p>
            </div>
            <div className="flex items-center gap-2">
              <button
                type="button"
                onClick={handleClearGraph}
                disabled={(!hasGraphData && !isInitialLoading) || isClearing}
                className="inline-flex items-center gap-2 rounded-md border border-red-500/70 bg-red-50 px-3 py-1.5 text-sm font-medium text-red-600 shadow-sm transition hover:bg-red-100 disabled:cursor-not-allowed disabled:opacity-60"
              >
                {isClearing ? (
                  <Loader2 className="h-4 w-4 animate-spin" />
                ) : (
                  <Trash2 className="h-4 w-4" />
                )}
                {isClearing ? "Clearing..." : "Clear graph"}
              </button>
              <button
                type="button"
                onClick={handleRefresh}
                className="inline-flex items-center gap-2 rounded-md border border-input bg-background px-3 py-1.5 text-sm font-medium text-foreground shadow-sm transition hover:bg-muted"
              >
                <RefreshCw className="h-4 w-4" />
                Reset graph
              </button>
              <button
                type="button"
                onClick={handleExpandSelected}
                disabled={!selectedNodeId || expandingNodeId === selectedNodeId || selectedNodeIsPlaceholder}
                className="inline-flex items-center gap-2 rounded-md border border-primary bg-primary px-3 py-1.5 text-sm font-medium text-primary-foreground shadow-sm transition hover:brightness-105 disabled:cursor-not-allowed disabled:opacity-60"
              >
                <ArrowUpRight className="h-4 w-4" />
                Expand selected
              </button>
            </div>
          </div>

          <div ref={containerRef} className="relative h-[540px] w-full bg-background">
            {isInitialLoading ? (
              <div className="absolute inset-0 flex items-center justify-center">
                <Loader2 className="h-8 w-8 animate-spin text-primary" />
              </div>
            ) : null}

            {hasGraphData ? (
              <svg width={dimensions.width} height={dimensions.height} className="block">
                <defs>
                  <filter id="node-shadow" x="-50%" y="-50%" width="200%" height="200%">
                    <feDropShadow dx="0" dy="2" stdDeviation="4" floodOpacity="0.18" />
                  </filter>
                </defs>

                {edges.map((edge) => {
                  const source = graph.nodes[edge.source];
                  const target = graph.nodes[edge.target];
                  if (!source || !target) {
                    return null;
                  }
                  const sourcePos = positions[source.id];
                  const targetPos = positions[target.id];
                  if (!sourcePos || !targetPos) {
                    return null;
                  }
                  const strokeWidth = clamp(1 + Math.log(edge.weight + 1), 1.25, 4.5);
                  const color = EDGE_COLORS[edge.type];
                  const isActive = selectedNodeId && (edge.source === selectedNodeId || edge.target === selectedNodeId);
                  const midX = (sourcePos.x + targetPos.x) / 2;
                  const midY = (sourcePos.y + targetPos.y) / 2;
                  const label = formatRelationLabel(edge.type);
                  return (
                    <g key={edge.id} className="transition-opacity">
                      <line
                        x1={sourcePos.x}
                        y1={sourcePos.y}
                        x2={targetPos.x}
                        y2={targetPos.y}
                        stroke={color}
                        strokeWidth={strokeWidth}
                        strokeOpacity={isActive ? 0.75 : 0.35}
                      />
                      <text
                        x={midX}
                        y={midY - 6}
                        textAnchor="middle"
                        fill={isActive ? color : "#475569"}
                        stroke="white"
                        strokeWidth={isActive ? 2 : 1.5}
                        strokeOpacity={0.9}
                        fontSize="11"
                        fontWeight={600}
                        style={{ paintOrder: "stroke" }}
                      >
                        {label}
                      </text>
                    </g>
                  );
                })}

                {nodesWithPositions.map((node) => {
                  const isSelected = node.id === selectedNodeId;
                  const isNeighbor = neighborSet.has(node.id);
                  const placeholder = isPlaceholderMetadata(node.metadata ?? undefined);
                  const color = NODE_COLORS[node.type] ?? "#475569";
                  const radius = isSelected ? 18 : 14;
                  return (
                    <g key={node.id} transform={`translate(${node.x}, ${node.y})`}>
                      <circle
                        r={radius}
                        fill={color}
                        fillOpacity={placeholder ? 0.45 : isSelected ? 1 : isNeighbor ? 0.9 : 0.75}
                        stroke={placeholder ? "#94a3b8" : isSelected ? "#1f2937" : "#0f172a"}
                        strokeWidth={isSelected ? 3 : 2}
                        filter="url(#node-shadow)"
                        className={`${placeholder ? "cursor-default" : "cursor-pointer"} transition-opacity`}
                        onClick={() => {
                          if (!placeholder) {
                            handleNodeSelect(node.id);
                          }
                        }}
                      />
                      <text
                        x={0}
                        y={radius + 18}
                        textAnchor="middle"
                        className={`select-none font-semibold ${placeholder ? "text-slate-500" : "text-slate-900"}`}
                        style={{ fontSize: "12px" }}
                      >
                        {node.label}
                      </text>
                    </g>
                  );
                })}
              </svg>
            ) : (
              <div className="flex h-full items-center justify-center px-6 text-center text-sm text-muted-foreground">
                {isGraphCleared
                  ? "Knowledge graph has been cleared. Upload a new paper to rebuild it."
                  : "Adjust filters to load graph data."}
              </div>
            )}

            {expandingNodeId ? (
              <div className="absolute bottom-4 left-1/2 flex -translate-x-1/2 items-center gap-2 rounded-full border border-primary/40 bg-background/80 px-4 py-1.5 text-xs font-medium text-primary shadow">
                <Loader2 className="h-3.5 w-3.5 animate-spin" />
                Expanding neighborhood…
              </div>
            ) : null}
          </div>
        </div>

        <div className="flex flex-wrap items-center gap-4 rounded-lg border bg-card px-4 py-3 text-xs text-muted-foreground">
          {ALL_TYPES.map((type) => (
            <div key={type} className="flex items-center gap-2">
              <span
                className="inline-flex h-3.5 w-3.5 items-center justify-center rounded-full"
                style={{ backgroundColor: NODE_COLORS[type] }}
              />
              {formatTypeLabel(type)} nodes
            </div>
          ))}
          <div className="flex items-center gap-2">
            <span className="inline-flex h-3.5 w-8 items-center justify-center rounded-full bg-muted text-[10px] font-semibold uppercase text-muted-foreground">
              Edge
            </span>
            Edge color encodes relation type
          </div>
        </div>

        {graphSummary ? (
          <div className="rounded-lg border border-dashed bg-muted/40 px-4 py-3 text-xs text-muted-foreground">
            {graphSummary}
          </div>
        ) : null}
      </div>

      <aside className="space-y-4">
        <div className="rounded-lg border bg-card p-5 shadow-sm">
          <h3 className="text-base font-semibold text-foreground">Node details</h3>
          {selectedNode ? (
            <div className="mt-4 space-y-4 text-sm">
              <div>
                <p className="text-xs font-medium uppercase tracking-wide text-primary">Label</p>
                <p className="mt-1 text-lg font-semibold text-foreground">{selectedNode.label}</p>
              </div>

              <div className="grid gap-2 text-xs text-muted-foreground">
                <div className="flex items-center justify-between">
                  <span className="uppercase tracking-wide">Type</span>
                  <span className="rounded-full border border-primary/30 bg-primary/10 px-2 py-0.5 font-semibold text-primary">
                    {formatTypeLabel(selectedNode.type)}
                  </span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="uppercase tracking-wide">Used by</span>
                  <span className="font-semibold text-foreground">{selectedNode.paper_count.toLocaleString()} papers</span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="uppercase tracking-wide">Expanded</span>
                  <span className="font-semibold text-foreground">
                    {expandedNodes[selectedNode.id] ? "Yes" : "No"}
                  </span>
                </div>
              </div>

              {selectedNode.aliases && selectedNode.aliases.length > 0 ? (
                <div className="space-y-2">
                  <p className="text-xs font-medium uppercase tracking-wide text-muted-foreground">Aliases</p>
                  <div className="flex flex-wrap gap-1">
                    {selectedNode.aliases.map((alias) => (
                      <span key={alias} className="rounded-full bg-muted px-2 py-0.5 text-[11px] font-medium text-muted-foreground">
                        {alias}
                      </span>
                    ))}
                  </div>
                </div>
              ) : null}

              {papersByYear.length > 0 ? (
                <div className="space-y-1">
                  <p className="text-xs font-medium uppercase tracking-wide text-muted-foreground">Papers by year</p>
                  <ul className="space-y-1 text-xs text-muted-foreground">
                    {papersByYear.map((entry) => (
                      <li key={entry.year} className="flex items-center justify-between rounded-md border border-border/60 bg-muted/30 px-2 py-1">
                        <span className="font-medium text-foreground/70">{entry.year}</span>
                        <span className="text-foreground">{entry.paper_count.toLocaleString()} papers</span>
                      </li>
                    ))}
                  </ul>
                </div>
              ) : null}

              {bestOutcome ? (
                <div className="space-y-1">
                  <p className="text-xs font-medium uppercase tracking-wide text-muted-foreground">Best reported outcome</p>
                  <div className="rounded-md border border-primary/30 bg-primary/5 p-2 text-xs text-muted-foreground">
                    <p className="font-semibold text-foreground/80">{formatOutcomeSummary(bestOutcome) || "Reported outcome"}</p>
                    {bestOutcome.paper_title ? (
                      <p className="mt-1 text-[11px]">{bestOutcome.paper_title}</p>
                    ) : null}
                    {formatOutcomeDetails(bestOutcome) ? (
                      <p className="mt-1 text-[10px] uppercase tracking-wide">{formatOutcomeDetails(bestOutcome)}</p>
                    ) : null}
                  </div>
                </div>
              ) : null}

              {worstOutcome && (!bestOutcome || worstOutcome.paper_id !== bestOutcome.paper_id || worstOutcome.value_numeric !== bestOutcome.value_numeric || worstOutcome.value_text !== bestOutcome.value_text) ? (
                <div className="space-y-1">
                  <p className="text-xs font-medium uppercase tracking-wide text-muted-foreground">Worst reported outcome</p>
                  <div className="rounded-md border border-destructive/30 bg-destructive/5 p-2 text-xs text-muted-foreground">
                    <p className="font-semibold text-foreground/80">{formatOutcomeSummary(worstOutcome) || "Reported outcome"}</p>
                    {worstOutcome.paper_title ? (
                      <p className="mt-1 text-[11px]">{worstOutcome.paper_title}</p>
                    ) : null}
                    {formatOutcomeDetails(worstOutcome) ? (
                      <p className="mt-1 text-[10px] uppercase tracking-wide">{formatOutcomeDetails(worstOutcome)}</p>
                    ) : null}
                  </div>
                </div>
              ) : null}

              {nodeEdgeInsights.length > 0 ? (
                <div className="space-y-2">
                  <p className="text-xs font-medium uppercase tracking-wide text-muted-foreground">Edge insights</p>
                  <ul className="space-y-2 text-xs text-muted-foreground">
                    {nodeEdgeInsights.map((insight, index) => (
                      <li key={`${insight.paper_id}-${index}`} className="rounded-md border border-border/60 bg-muted/30 p-2">
                        <p className="font-semibold text-foreground/80">{insight.summary}</p>
                        <p className="mt-1 text-[10px] uppercase tracking-wide text-muted-foreground">
                          {formatRelationLabel(insight.relation)} • {insight.paper_year ?? "Year unknown"}
                        </p>
                        {insight.paper_title ? (
                          <p className="mt-1 text-[11px] text-muted-foreground/90">{insight.paper_title}</p>
                        ) : null}
                      </li>
                    ))}
                  </ul>
                </div>
              ) : null}

              {remainingMetadataEntries.length > 0 ? (
                <div className="space-y-2">
                  <p className="text-xs font-medium uppercase tracking-wide text-muted-foreground">Metadata</p>
                  <dl className="grid gap-1 text-xs text-muted-foreground">
                    {remainingMetadataEntries.map(([key, value]) => (
                      <div key={key} className="flex justify-between gap-2">
                        <dt className="font-medium capitalize text-foreground/70">{key}</dt>
                        <dd className="text-right">{formatMetadataValue(value)}</dd>
                      </div>
                    ))}
                  </dl>
                </div>
              ) : null}

              {selectedNode.top_links.length > 0 ? (
                <div className="space-y-2">
                  <p className="text-xs font-medium uppercase tracking-wide text-muted-foreground">Top linked nodes</p>
                  <ul className="space-y-1 text-xs text-muted-foreground">
                    {selectedNode.top_links.map((link) => (
                      <li
                        key={link.id}
                        className="flex items-center justify-between rounded-md border border-border/60 bg-muted/30 px-2 py-1"
                      >
                        <div className="flex flex-col">
                          <span className="font-semibold text-foreground/80">{link.label}</span>
                          <span className="text-[10px] uppercase tracking-wide text-muted-foreground">
                            {formatRelationLabel(link.relation)}
                          </span>
                        </div>
                        <div className="text-right">
                          <span className="rounded-full bg-background px-2 py-0.5 text-[10px] font-semibold uppercase tracking-wide text-muted-foreground">
                            {formatTypeLabel(link.type)}
                          </span>
                          <div className="text-[10px] font-medium text-muted-foreground">
                            Weight {link.weight.toFixed(2)}
                          </div>
                        </div>
                      </li>
                    ))}
                  </ul>
                </div>
              ) : (
                <p className="text-xs text-muted-foreground">Expand the node to reveal its strongest connections.</p>
              )}

              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <p className="text-xs font-medium uppercase tracking-wide text-muted-foreground">Evidence</p>
                  <button
                    type="button"
                    onClick={() => setShowEvidence((current) => !current)}
                    className="text-xs font-medium text-primary underline-offset-2 hover:underline"
                  >
                    {showEvidence ? "Hide" : "Why?"}
                  </button>
                </div>
                {showEvidence ? (
                  selectedNode.evidence.length > 0 ? (
                    <ul className="space-y-2 text-xs text-muted-foreground">
                      {selectedNode.evidence.map((item, index) => (
                        <li key={`${item.paper_id}-${index}`} className="rounded-md border border-border/60 bg-muted/30 p-2">
                          <div className="flex items-center justify-between">
                            <span className="font-semibold text-foreground/80">{item.paper_title ?? "Unknown paper"}</span>
                            <span className="text-[10px] uppercase tracking-wide text-muted-foreground">
                              {formatRelationLabel(item.relation)} • {(item.confidence * 100).toFixed(0)}%
                            </span>
                          </div>
                          {item.snippet ? <p className="mt-1 text-[13px] leading-snug text-foreground/80">“{item.snippet}”</p> : null}
                        </li>
                      ))}
                    </ul>
                  ) : (
                    <p className="text-xs text-muted-foreground">Evidence snippets will appear after expanding connected edges.</p>
                  )
                ) : null}
              </div>
            </div>
          ) : (
            <p className="mt-3 text-sm text-muted-foreground">Select a node in the graph to view its metadata and connections.</p>
          )}
        </div>

        <div className="rounded-lg border bg-card p-5 text-sm text-muted-foreground">
          <h3 className="text-base font-semibold text-foreground">How to explore</h3>
          <ul className="mt-3 space-y-2 text-sm">
            <li>Toggle entity types and relation categories to focus on specific portions of the graph.</li>
            <li>Adjust the confidence slider to hide uncertain relationships and highlight stronger evidence.</li>
            <li>Click any node to load its neighborhood and inspect the top supporting papers via the “Why?” button.</li>
          </ul>
        </div>
      </aside>
    </div>
  );
};

export default GraphExplorer;

