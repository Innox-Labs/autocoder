"""AutoCoder prototype: outpatient clinical note -> ICD-10-CM diagnosis codes.

Pipeline stages (see prototype design doc, Section 4):
  1. intake      -> NormalizedDocument
  2. sections    -> list[Section]
  3. ner         -> raw entity spans (OpenMed models)
  4. assertion   -> list[CodableFact] (OpenMed context + outpatient filter)
  5. retrieval   -> candidates per fact (lexical + semantic, versioned terminology)
  6. reasoner    -> proposed CodeAssignments (LLM, candidate-constrained)
  7. rules       -> deterministic validation + confidence + routing
"""

__version__ = "0.1.0"
