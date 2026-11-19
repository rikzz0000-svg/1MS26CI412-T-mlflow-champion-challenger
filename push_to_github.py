#!/usr/bin/env python3
"""
push_to_github.py
Helper script to push this initialized local git repository directly to GitHub.

Usage:
  python push_to_github.py <YOUR_GITHUB_PERSONAL_ACCESS_TOKEN>
or
  set GITHUB_TOKEN=<YOUR_TOKEN>
  python push_to_github.py
"""

import os
import sys
from dulwich import porcelain

REPO_PATH = os.path.dirname(os.path.abspath(__file__))
REMOTE_REPO = "github.com/rikzz0000-svg/1MS26CI412-T-mlflow-champion-challenger.git"

def main():
    token = None
    if len(sys.argv) > 1:
        token = sys.argv[1].strip()
    elif "GITHUB_TOKEN" in os.environ:
        token = os.environ["GITHUB_TOKEN"].strip()
    else:
        print("Please enter your GitHub Personal Access Token (PAT) with repo permissions:")
        try:
            token = input("GitHub Token: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nAborted.")
            sys.exit(1)

    if not token:
        print("[ERROR] Token cannot be empty.")
        sys.exit(1)

    # Format authenticated URL
    authenticated_url = f"https://{token}@{REMOTE_REPO}"
    print(f"[Git] Pushing branch 'main' to https://{REMOTE_REPO}...")
    try:
        porcelain.push(REPO_PATH, authenticated_url, "refs/heads/main:refs/heads/main")
        print("\n[SUCCESS] Successfully pushed all 3 commits and files to GitHub!")
        print(f"View your repository at: https://{REMOTE_REPO}")
    except Exception as e:
        print(f"\n[ERROR] Git push failed: {e}")
        print("Please ensure your token has 'repo' write permissions and that the repository exists on GitHub.")

if __name__ == "__main__":
    main()
