from memory_store import store
from state import AgentState

def memory_retrieval(state:AgentState)->AgentState:
    user_id = state.get("user_id", "default_user")
    query = state["user_query"]
    #-----Semantic memory for all users    
    semantic_results = store.search(("semantic",),query=query, limit = 5)
    semantic_facts = [r.value.get("content","") for r in semantic_results]

    #-----Episodic memory for particular user
    episodic_results = store.search(("episodes",user_id),query=query,limit=3)
    past_episodes = [r.value.get("content","") for r in episodic_results]

    #----- Procudral memory for the behavioural study for individual user
    procedural_results = store.search(("procedural",user_id),query="", limit=50)
    procedural_rules = [r.value.get("content","") for r in procedural_results]

    return {**state, "semantic_facts": semantic_facts,
             "past_episodes": past_episodes,
             "procedural_rules":procedural_rules}

