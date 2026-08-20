# RAG evaluation

Context adherence measures whether an answer is supported by retrieved context.
Chunk attribution identifies which retrieved passages support an answer, while
chunk utilization measures how much of the supplied context was useful.
Completeness measures whether the answer covers the information requested by the
user. These evaluations require a retrieval step with document content in its
telemetry span.