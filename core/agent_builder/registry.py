"""Agent builder registry."""

from typing import List
from typing import Union
from pathlib import Path
import json
import shutil

from core.param_cache import ParamCache


class AgentCacheRegistry:
    """Registry for agent caches, in disk.

    Can register new agent caches, load agent caches, delete agent caches, etc.

    """

    def __init__(self, dir: Union[str, Path]) -> None:
        """Init params."""
        self._dir = dir

    def _add_agent_id_to_directory(self, agent_id: str) -> None:
        """Save agent id to directory."""
        full_path = Path(self._dir) / "agent_ids.json"
        if not full_path.exists():
            with open(full_path, "w") as f:
                json.dump({"agent_ids": [agent_id]}, f)
        else:
            with open(full_path, "r") as f:
                agent_ids = json.load(f)["agent_ids"]
            if agent_id in agent_ids:
                raise ValueError(f"Agent id {agent_id} already exists.")
            agent_ids_set = set(agent_ids)
            agent_ids_set.add(agent_id)
            with open(full_path, "w") as f:
                json.dump({"agent_ids": list(agent_ids_set)}, f)

    def add_new_agent_cache(self, agent_id: str, cache: ParamCache) -> None:
        """Register agent."""
        # save the cache to disk
        agent_cache_path = f"{self._dir}/{agent_id}"
        cache.save_to_disk(agent_cache_path)
        # save to agent ids
        self._add_agent_id_to_directory(agent_id)

    def get_agent_ids(self) -> List[str]:
        """Get agent ids."""
        full_path = Path(self._dir) / "agent_ids.json"
        if not full_path.exists():
            return []
        with open(full_path, "r") as f:
            agent_ids = json.load(f)["agent_ids"]

        return agent_ids

    def get_agent_cache(self, agent_id: str) -> ParamCache:
        """Get agent cache."""
        full_path = Path(self._dir) / f"{agent_id}"
        if not full_path.exists():
            raise ValueError(f"Cache for agent {agent_id} does not exist.")
        cache = ParamCache.load_from_disk(str(full_path))
        return cache

    def delete_agent_cache(self, agent_id: str) -> None:
        """Delete agent cache."""
        # modify / resave agent_ids
        agent_ids = self.get_agent_ids()
        new_agent_ids = [id for id in agent_ids if id != agent_id]
        full_path = Path(self._dir) / "agent_ids.json"
        with open(full_path, "w") as f:
            json.dump({"agent_ids": new_agent_ids}, f)

        # remove agent cache
        full_path = Path(self._dir) / f"{agent_id}"
        if full_path.exists():
            # recursive delete
            shutil.rmtree(full_path)
