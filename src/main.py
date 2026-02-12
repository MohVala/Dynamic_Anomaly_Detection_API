from core.pipeline import run_pipeline
from core.context import RunContext

if __name__ == "__main__":
    context = RunContext(env="dev")

    run_pipeline(context=context)