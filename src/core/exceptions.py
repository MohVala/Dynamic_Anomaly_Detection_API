class PipelineError(Exception):
    """Base exception for pipeline errors."""

class IngestionError(PipelineError):
    """Raised when ingestion fails."""

class ProcessingError(PipelineError):
    """Raised when processing fails."""

class ModellingError(PipelineError):
    """Raised when modelling fails."""

class ReportingError(PipelineError):
    """Raised when reporting fails."""

class ValidationError(PipelineError):
    """Raised when schema or statistal validation fails."""