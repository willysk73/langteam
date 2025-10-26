"""Test script for the multiagent system."""
import os
from dotenv import load_dotenv
from multiagent_supervisor import MultiAgentSystem

# Load environment variables
load_dotenv()


def test_basic_workflow():
    """Test basic multiagent workflow."""
    print("=" * 60)
    print("Testing Multiagent System")
    print("=" * 60)
    
    # Check API key
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Error: OPENAI_API_KEY not found in .env file")
        return
    
    print("✅ API key found")
    
    # Initialize system
    print("\n🔧 Initializing multiagent system...")
    try:
        system = MultiAgentSystem()
        print("✅ System initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize system: {e}")
        return
    
    # Test simple task that requires multiple agents
    test_task = "Calculate 15 * 8 and then write a brief summary of the result"
    
    print(f"\n📝 Running test task: {test_task}")
    print("-" * 60)
    
    try:
        result = system.run(test_task)
        
        print("\n" + "=" * 60)
        print("✅ TEST PASSED")
        print("=" * 60)
        print("\n📊 Task Results:")
        for agent_name, output in result.get("task_result", {}).items():
            print(f"\n{agent_name}:")
            print(f"  {output}")
        
        print(f"\n💬 Total messages exchanged: {len(result.get('messages', []))}")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_basic_workflow()
