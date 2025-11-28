#!/usr/bin/env python3
"""
Script to create a GitHub repository and connect it to the local git repo.
Requires a GitHub Personal Access Token.
"""

import subprocess
import sys
import json
import os
from pathlib import Path

def run_command(cmd, check=True):
    """Run a shell command."""
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if check and result.returncode != 0:
        print(f"Error: {result.stderr}")
        sys.exit(1)
    return result

def check_git_remote():
    """Check if remote is already configured."""
    result = run_command("git remote -v", check=False)
    if result.returncode == 0 and result.stdout.strip():
        print("Current remotes:")
        print(result.stdout)
        response = input("\nRemote already exists. Overwrite? (y/n): ")
        if response.lower() != 'y':
            return False
        run_command("git remote remove origin", check=False)
    return True

def create_github_repo(token, repo_name, description, private=False):
    """Create a GitHub repository using the API."""
    import urllib.request
    import urllib.error
    
    url = "https://api.github.com/user/repos"
    data = {
        "name": repo_name,
        "description": description,
        "private": private,
        "auto_init": False  # We already have files
    }
    
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json",
        "Content-Type": "application/json"
    }
    
    req = urllib.request.Request(url, data=json.dumps(data).encode(), headers=headers)
    
    try:
        with urllib.request.urlopen(req) as response:
            result = json.loads(response.read().decode())
            return result.get("clone_url"), result.get("ssh_url")
    except urllib.error.HTTPError as e:
        error_msg = e.read().decode()
        print(f"Error creating repository: {error_msg}")
        if e.code == 401:
            print("\nAuthentication failed. Please check your token.")
        elif e.code == 422:
            print("\nRepository might already exist or name is invalid.")
        sys.exit(1)

def main():
    print("=" * 60)
    print("GitHub Repository Setup")
    print("=" * 60)
    print()
    
    # Get repository name
    repo_name = input("Repository name [Segmentation-detection]: ").strip()
    if not repo_name:
        repo_name = "Segmentation-detection"
    
    # Get description
    description = input("Description [Point cloud semantic segmentation using PointNet++]: ").strip()
    if not description:
        description = "Point cloud semantic segmentation using PointNet++"
    
    # Get visibility
    private_input = input("Private repository? (y/n) [n]: ").strip().lower()
    is_private = private_input == 'y'
    
    # Get GitHub token
    print("\nYou need a GitHub Personal Access Token.")
    print("Create one at: https://github.com/settings/tokens")
    print("Required scopes: 'repo' (full control of private repositories)")
    print()
    token = input("GitHub Personal Access Token: ").strip()
    
    if not token:
        print("Token is required. Exiting.")
        sys.exit(1)
    
    # Check if we're in a git repo
    if not Path(".git").exists():
        print("Error: Not in a git repository. Run 'git init' first.")
        sys.exit(1)
    
    # Check remote
    if not check_git_remote():
        print("Aborted.")
        sys.exit(0)
    
    # Create repository on GitHub
    print(f"\nCreating repository '{repo_name}' on GitHub...")
    try:
        https_url, ssh_url = create_github_repo(token, repo_name, description, is_private)
        print(f"✓ Repository created successfully!")
        print(f"  HTTPS: {https_url}")
        print(f"  SSH: {ssh_url}")
    except ImportError:
        print("Error: urllib not available. Please install Python standard library.")
        sys.exit(1)
    
    # Add remote
    print(f"\nAdding remote 'origin'...")
    run_command(f'git remote add origin {https_url}')
    print("✓ Remote added")
    
    # Push to GitHub
    print(f"\nPushing to GitHub...")
    run_command("git branch -M main")
    run_command("git push -u origin main")
    print("✓ Pushed to GitHub!")
    
    print("\n" + "=" * 60)
    print("✓ Repository setup complete!")
    print("=" * 60)
    print(f"\nRepository URL: {https_url}")
    print(f"You can view it at: https://github.com/{repo_name.split('/')[-1] if '/' in repo_name else 'YOUR_USERNAME'}/{repo_name}")

if __name__ == "__main__":
    main()

