"""
Authentication helper for Hugging Face datasets
"""

import os
from huggingface_hub import login, whoami
from datasetCPICANN_loader import DatasetCPICANNLoader


def check_auth_status():
    """
    Check if user is authenticated with Hugging Face
    """
    try:
        user_info = whoami()
        print(f"✅ Authenticated as: {user_info['name']}")
        return True
    except Exception as e:
        print(f"❌ Not authenticated: {e}")
        return False


def authenticate_with_token(token: str):
    """
    Authenticate using a token
    
    Args:
        token: Hugging Face token
    """
    try:
        login(token=token)
        print("✅ Authentication successful!")
        return True
    except Exception as e:
        print(f"❌ Authentication failed: {e}")
        return False


def authenticate_interactive():
    """
    Interactive authentication
    """
    print("Hugging Face Authentication")
    print("=" * 30)
    
    # Check current status
    if check_auth_status():
        return True
    
    print("\nTo authenticate, you can:")
    print("1. Use a token directly")
    print("2. Use environment variable")
    print("3. Use huggingface-cli login")
    
    choice = input("\nEnter your choice (1-3): ").strip()
    
    if choice == "1":
        token = input("Enter your Hugging Face token: ").strip()
        return authenticate_with_token(token)
    elif choice == "2":
        print("Set environment variable: export HUGGINGFACE_HUB_TOKEN='your_token'")
        print("Then restart your Python session.")
        return False
    elif choice == "3":
        print("Run: huggingface-cli login")
        print("Then restart your Python session.")
        return False
    else:
        print("Invalid choice")
        return False


def test_dataset_access():
    """
    Test access to the datasetCPICANN dataset
    """
    print("\nTesting dataset access...")
    
    try:
        loader = DatasetCPICANNLoader()
        
        # Try to get repository info first
        from huggingface_hub import HfApi
        api = HfApi()
        
        try:
            repo_info = api.repo_info("caobin/datasetCPICANN", repo_type="dataset")
            print(f"✅ Repository found: {repo_info.id}")
            print(f"   Private: {repo_info.private}")
            print(f"   Gated: {getattr(repo_info, 'gated', False)}")
            
            if repo_info.private or getattr(repo_info, 'gated', False):
                print("⚠️  This is a private/gated dataset. Authentication required.")
            else:
                print("✅ This is a public dataset.")
                
        except Exception as e:
            print(f"❌ Repository access error: {e}")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing dataset access: {e}")
        return False


def main():
    """
    Main authentication helper
    """
    print("Hugging Face Dataset Authentication Helper")
    print("=" * 40)
    
    # Test dataset access first
    if not test_dataset_access():
        print("\n❌ Cannot access dataset. Please check:")
        print("1. Repository name is correct")
        print("2. You have access permissions")
        print("3. Repository exists")
        return False
    
    # Check authentication
    if check_auth_status():
        print("\n✅ Ready to load dataset!")
        return True
    
    # Try to authenticate
    print("\n🔐 Authentication required")
    if authenticate_interactive():
        print("\n✅ Ready to load dataset!")
        return True
    else:
        print("\n❌ Authentication failed. Please try again.")
        return False


if __name__ == "__main__":
    success = main()
    if success:
        print("\nYou can now run:")
        print("python example_usage.py")
    else:
        print("\nPlease resolve authentication issues before proceeding.")
