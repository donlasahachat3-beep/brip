"""Entry point for running the agent manager API."""

import uvicorn


def main() -> None:
    uvicorn.run("core.agent_manager.api:app", host="0.0.0.0", port=8100, reload=False)


if __name__ == "__main__":
    main()
