# Claude Code PR Comment Resolution System Prompt

You are using Claude Code to systematically resolve all comments, to-dos, and issues in a pull request. Claude Code operates directly in your terminal, understands context, maintains awareness of your entire project structure, and takes action by performing real operations like editing files and creating commits.

## Context Awareness

Claude Code can work with PR context in two ways:

1. **Automatic Detection**: If you're on a PR branch, Claude Code will automatically detect the current branch and associated PR context
2. **Manual Specification**: You can specify a PR number through `$ARGUMENTS` to work on a specific PR

Claude Code will:
- Use the PR number from `$ARGUMENTS` if provided
- Otherwise, detect the current branch and associated PR
- See all PR comments and review threads automatically

## Workflow Approach

Follow the research/plan/implement pattern that significantly improves performance for problems requiring deeper thinking:

### Phase 1: Research & Analysis

```
Please analyze this PR and all its comments comprehensively. Look for:
1. All unresolved review comments and conversations
2. To-do items mentioned in comments  
3. Requested changes from code reviews
4. Questions that need responses

Use the GitHub API systematically to get complete data about all comment types.
```

### Phase 2: Planning

```
Based on your analysis, create a detailed plan to address all unresolved items.
Group them by type (code changes, documentation, responses to questions).
Prioritize based on importance and dependencies.
Use TodoWrite tool to track progress systematically.
```

### Phase 3: Implementation

```
Now implement the solutions for each item in the plan:
- Make the requested code changes
- Update documentation as needed
- Prepare responses to questions
- Ensure all changes maintain code quality and pass tests
- Mark each todo as completed when finished
```

### Phase 4: Resolution & Verification

```
After addressing all items:
1. Run linting and tests to verify everything works
2. Create a summary of all changes made
3. Commit the changes with a clear message
4. Verify all review comments have been addressed
```

## Correct GitHub API Commands for Comment Retrieval

The key is to use the right API endpoints that actually work. Here are the tested, working commands:

### Prerequisites & Error Handling

```bash
# Check prerequisites and API rate limits
if ! command -v gh &> /dev/null; then
    echo "Error: GitHub CLI 'gh' is not installed"
    exit 1
fi

if ! command -v jq &> /dev/null; then
    echo "Error: 'jq' is not installed"
    exit 1
fi

# Check GitHub authentication
if ! gh auth status &> /dev/null; then
    echo "Error: Not authenticated with GitHub. Run 'gh auth login'"
    exit 1
fi

# Check API rate limit
RATE_LIMIT=$(gh api rate_limit | jq -r '.resources.core.remaining')
if [ "$RATE_LIMIT" -lt 10 ]; then
    echo "Warning: GitHub API rate limit low ($RATE_LIMIT remaining)"
fi
```

### 1. Get Current PR Number and Basic Info
```bash
# Get current PR details
# If PR number is provided in $ARGUMENTS, use it. Otherwise detect from current branch
if [ -n "$ARGUMENTS" ]; then
    PR_NUM="$ARGUMENTS"
    echo "Using specified PR #$PR_NUM"
else
    PR_NUM=$(gh pr view --json number | jq -r '.number')
    echo "Detected PR #$PR_NUM from current branch"
fi

# View PR with all comments (readable format)
gh pr view $PR_NUM --comments
```

### 2. Get All Review Comments (Code-Level Comments)
```bash
# Get review comments (comments on specific lines of code)
gh api repos/{owner}/{repo}/pulls/{pr_number}/comments 2>/dev/null | jq '.[] | {id: .id, author: .author.login, body: .body, created_at: .created_at, in_reply_to_id: .in_reply_to_id}' 2>/dev/null || echo "Failed to fetch comments"

# For current repo, use variables (reliable approach with error handling):
# Try with -q flag first (more reliable)
if ! OWNER=$(gh repo view --json owner -q .owner.login 2>/dev/null); then
    # Fallback to jq method
    if ! OWNER=$(gh repo view --json owner 2>/dev/null | jq -r '.owner.login' 2>/dev/null); then
        echo "Error: Unable to get repository owner"
        echo "Tip: Make sure you're in a git repository and authenticated with GitHub"
        exit 1
    fi
fi

if ! REPO=$(gh repo view --json name -q .name 2>/dev/null); then
    # Fallback to jq method
    if ! REPO=$(gh repo view --json name 2>/dev/null | jq -r '.name' 2>/dev/null); then
        echo "Error: Unable to get repository name"
        exit 1
    fi
fi

echo "Repository detected: $OWNER/$REPO"

if [ -n "$ARGUMENTS" ]; then
    PR_NUM="$ARGUMENTS"
    echo "Using specified PR #$PR_NUM"
elif ! PR_NUM=$(gh pr view --json number | jq -r '.number' 2>/dev/null); then
    echo "Error: Unable to get PR number. Are you on a PR branch?"
    exit 1
fi

echo "Fetching comments for PR #$PR_NUM in $OWNER/$REPO"
gh api repos/$OWNER/$REPO/pulls/$PR_NUM/comments
```

### 3. Get All Review Summaries
```bash
# Get review summaries (approve/request changes/comment)
gh api repos/{owner}/{repo}/pulls/{pr_number}/reviews | jq '.[] | select(.state == "COMMENTED") | {id: .id, author: .author.login, body: .body, submitted_at: .submitted_at}'
```

### 4. Get Issue Comments (General PR Comments)
```bash
# Get general PR discussion comments with error handling
gh pr view --json comments 2>/dev/null | jq '.comments[] | {id: .id, author: .author.login, body: .body, created_at: .created_at}' 2>/dev/null || echo "Failed to fetch PR comments"
```

### 5. Reply to Review Comments and Mark as Resolved (Hybrid Method - Recommended)

```bash
# Method 1: GraphQL-based review thread reply and resolve (Most Reliable)
reply_and_resolve_by_text() {
    local SEARCH_TEXT="$1"
    local REPLY_BODY="$2"
    
    echo "🔍 Searching for comment containing: $SEARCH_TEXT"
    
    # Ensure OWNER and REPO are set
    if [[ -z "$OWNER" || -z "$REPO" ]]; then
        echo "⚠️ Setting OWNER and REPO manually..."
        OWNER=$(gh repo view --json owner -q .owner.login 2>/dev/null || echo "")
        REPO=$(gh repo view --json name -q .name 2>/dev/null || echo "")
        
        if [[ -z "$OWNER" || -z "$REPO" ]]; then
            echo "❌ Failed to detect repository. Please set OWNER and REPO variables."
            return 1
        fi
    fi
    
    # Get all review threads for the PR with control character handling
    echo "Fetching review threads..."
    local GRAPHQL_RESPONSE=$(gh api graphql -f query="
    {
      repository(owner: \"$OWNER\", name: \"$REPO\") {
        pullRequest(number: $PR_NUM) {
          reviewThreads(first: 100) {
            nodes {
              id
              isResolved
              comments(first: 10) {
                nodes {
                  id
                  body
                  author { login }
                }
              }
            }
          }
        }
      }
    }" 2>/dev/null)
    
    if [ $? -eq 0 ] && [ -n "$GRAPHQL_RESPONSE" ]; then
        # Clean control characters and parse JSON
        local THREADS=$(echo "$GRAPHQL_RESPONSE" | tr -d '\000-\031' | jq '.data.repository.pullRequest.reviewThreads.nodes' 2>/dev/null || echo "[]")
    else
        echo "Failed to fetch GraphQL data"
        local THREADS="[]"
    fi
    
    # Find thread and comment containing the search text with safe parsing
    local THREAD_ID=""
    local COMMENT_ID=""
    
    if [ "$THREADS" != "[]" ] && [ -n "$THREADS" ]; then
        THREAD_ID=$(echo "$THREADS" | jq -r \
          --arg text "$SEARCH_TEXT" \
          '.[] | select(.comments.nodes[].body | test($text; "i")) | .id' 2>/dev/null | head -1)
        
        COMMENT_ID=$(echo "$THREADS" | jq -r \
          --arg text "$SEARCH_TEXT" \
          '.[] | select(.comments.nodes[].body | test($text; "i")) | .comments.nodes[0].id' 2>/dev/null | head -1)
    fi
    
    if [[ -z "$THREAD_ID" || -z "$COMMENT_ID" ]]; then
        echo "❌ Could not find comment containing: $SEARCH_TEXT"
        return 1
    fi
    
    echo "💬 Found Comment ID: $COMMENT_ID"
    echo "🧵 Found Thread ID: $THREAD_ID"
    
    # Add reply to the review thread with better error handling
    echo "Adding reply..."
    local REPLY_RESULT=$(gh api graphql -f query="
    mutation {
      addPullRequestReviewThreadReply(input: {
        pullRequestReviewThreadId: \"$THREAD_ID\"
        body: \"$REPLY_BODY\"
      }) {
        comment {
          id
          body
        }
      }
    }" 2>&1)
    
    if echo "$REPLY_RESULT" | grep -q '"id"'; then
        echo "✅ Reply added successfully"
    else
        echo "❌ Failed to add reply: $REPLY_RESULT"
        return 1
    fi
    
    # Resolve the thread with better error handling
    echo "Resolving thread..."
    local RESOLVE_RESULT=$(gh api graphql -f query="
    mutation {
      resolveReviewThread(input: { threadId: \"$THREAD_ID\" }) {
        thread {
          isResolved
        }
      }
    }" 2>&1)
    
    if echo "$RESOLVE_RESULT" | grep -q '"isResolved"'; then
        echo "✅ Thread resolved successfully"
    else
        echo "⚠️ Thread resolution failed: $RESOLVE_RESULT"
    fi
}

# Method 2: Reply by specific comment ID
reply_and_resolve_by_id() {
    local COMMENT_ID="$1"
    local REPLY_BODY="$2"
    
    echo "🔍 Finding thread for comment ID: $COMMENT_ID"
    
    # Ensure OWNER and REPO are set
    if [[ -z "$OWNER" || -z "$REPO" ]]; then
        echo "⚠️ Setting OWNER and REPO manually..."
        OWNER=$(gh repo view --json owner -q .owner.login 2>/dev/null || echo "")
        REPO=$(gh repo view --json name -q .name 2>/dev/null || echo "")
        
        if [[ -z "$OWNER" || -z "$REPO" ]]; then
            echo "❌ Failed to detect repository. Please set OWNER and REPO variables."
            return 1
        fi
    fi
    
    # Get all review threads and find the one containing this comment
    echo "Fetching review threads..."
    local GRAPHQL_RESPONSE=$(gh api graphql -f query="
    {
      repository(owner: \"$OWNER\", name: \"$REPO\") {
        pullRequest(number: $PR_NUM) {
          reviewThreads(first: 100) {
            nodes {
              id
              isResolved
              comments(first: 10) {
                nodes {
                  id
                  body
                  author { login }
                }
              }
            }
          }
        }
      }
    }" 2>/dev/null)
    
    if [ $? -eq 0 ] && [ -n "$GRAPHQL_RESPONSE" ]; then
        # Clean control characters and parse JSON
        local THREADS=$(echo "$GRAPHQL_RESPONSE" | tr -d '\000-\031' | jq '.data.repository.pullRequest.reviewThreads.nodes' 2>/dev/null || echo "[]")
    else
        echo "Failed to fetch GraphQL data"
        local THREADS="[]"
    fi
    
    local THREAD_ID=""
    if [ "$THREADS" != "[]" ] && [ -n "$THREADS" ]; then
        THREAD_ID=$(echo "$THREADS" | jq -r \
          --arg id "$COMMENT_ID" \
          '.[] | select(.comments.nodes[].id == $id) | .id' 2>/dev/null | head -1)
    fi
    
    if [[ -z "$THREAD_ID" ]]; then
        echo "❌ Could not find thread for comment ID: $COMMENT_ID"
        return 1
    fi
    
    echo "🧵 Found Thread ID: $THREAD_ID"
    
    # Add reply to the review thread with better error handling
    echo "Adding reply..."
    local REPLY_RESULT=$(gh api graphql -f query="
    mutation {
      addPullRequestReviewThreadReply(input: {
        pullRequestReviewThreadId: \"$THREAD_ID\"
        body: \"$REPLY_BODY\"
      }) {
        comment {
          id
          body
        }
      }
    }" 2>&1)
    
    if echo "$REPLY_RESULT" | grep -q '"id"'; then
        echo "✅ Reply added successfully"
    else
        echo "❌ Failed to add reply: $REPLY_RESULT"
        return 1
    fi
    
    # Resolve the thread with better error handling
    echo "Resolving thread..."
    local RESOLVE_RESULT=$(gh api graphql -f query="
    mutation {
      resolveReviewThread(input: { threadId: \"$THREAD_ID\" }) {
        thread {
          isResolved
        }
      }
    }" 2>&1)
    
    if echo "$RESOLVE_RESULT" | grep -q '"isResolved"'; then
        echo "✅ Thread resolved successfully"
    else
        echo "⚠️ Thread resolution failed: $RESOLVE_RESULT"
    fi
}

# Method 3: Fallback REST API method (Legacy, less reliable)
reply_to_comment_rest() {
    local COMMENT_ID=$1
    local REPLY_MESSAGE=$2
    
    echo "Attempting REST API reply to comment $COMMENT_ID..."
    
    # Try direct reply first
    if gh api repos/$OWNER/$REPO/pulls/$PR_NUM/comments/$COMMENT_ID/replies \
        --method POST \
        -f body="$REPLY_MESSAGE" 2>/dev/null; then
        echo "✅ Direct reply successful"
    else
        echo "⚠️ Direct reply failed, adding as general PR comment..."
        gh pr comment $PR_NUM --body "**코멘트 ID $COMMENT_ID에 대한 답변:**
$REPLY_MESSAGE"
        echo "✅ Added as general PR comment"
    fi
}
```

### 6. Filter for Unresolved Comments Based on Content Analysis

Since GitHub doesn't have a "resolved" status for individual comments, you need to analyze content:

```bash
# Look for comments that indicate unresolved issues
gh api repos/{owner}/{repo}/pulls/{pr_number}/comments | jq '.[] | select(.body | test("(todo|TODO|FIXME|suggestion|issue|problem|fix|change|update|add|remove|modify)"; "i")) | {id: .id, body: .body, author: .author.login}'
```

### 7. Advanced: Check for Comments Without Follow-up Responses

```bash
# Get all comments and check which ones don't have replies
gh api repos/{owner}/{repo}/pulls/{pr_number}/comments | jq 'group_by(.in_reply_to_id // .id) | map(select(length == 1 and .[0].in_reply_to_id == null)) | flatten | .[] | {id: .id, body: .body, needs_response: true}'
```

## Practical Comment Resolution Workflow

### Step 1: Comprehensive Comment Discovery
```bash
# Set up variables first with error handling
# Try with -q flag first (more reliable)
if ! OWNER=$(gh repo view --json owner -q .owner.login 2>/dev/null); then
    # Fallback to jq method
    if ! OWNER=$(gh repo view --json owner 2>/dev/null | jq -r '.owner.login' 2>/dev/null); then
        echo "Error: Unable to get repository owner"
        echo "Tip: Make sure you're in a git repository and authenticated with GitHub"
        exit 1
    fi
fi

if ! REPO=$(gh repo view --json name -q .name 2>/dev/null); then
    # Fallback to jq method
    if ! REPO=$(gh repo view --json name 2>/dev/null | jq -r '.name' 2>/dev/null); then
        echo "Error: Unable to get repository name"
        exit 1
    fi
fi

if [ -n "$ARGUMENTS" ]; then
    PR_NUM="$ARGUMENTS"
    echo "Using specified PR #$PR_NUM"
elif ! PR_NUM=$(gh pr view --json number | jq -r '.number' 2>/dev/null); then
    echo "Error: Unable to get PR number. Are you on a PR branch?"
    exit 1
fi

echo "Processing comments for PR #$PR_NUM in $OWNER/$REPO"

# Run all these in parallel for complete picture:
gh pr view --comments  # Human-readable overview

# Get review comments with error handling
if COMMENTS_DATA=$(gh api repos/$OWNER/$REPO/pulls/$PR_NUM/comments 2>/dev/null); then
    echo "$COMMENTS_DATA" | jq '.[] | {id: .id, author: .author.login, body: .body, created_at: .created_at, in_reply_to_id: .in_reply_to_id}'
else
    echo "⚠️ Failed to fetch review comments"
fi

# Get review summaries with error handling
if REVIEWS_DATA=$(gh api repos/$OWNER/$REPO/pulls/$PR_NUM/reviews 2>/dev/null); then
    echo "$REVIEWS_DATA" | jq '.[] | select(.state == "COMMENTED") | {id: .id, author: .author.login, body: .body, submitted_at: .submitted_at}'
else
    echo "⚠️ Failed to fetch review summaries"
fi
```

### Step 2: Intelligent Comment Classification
```bash
# Set up variables first with error handling
# Try with -q flag first (more reliable)
if ! OWNER=$(gh repo view --json owner -q .owner.login 2>/dev/null); then
    # Fallback to jq method
    if ! OWNER=$(gh repo view --json owner 2>/dev/null | jq -r '.owner.login' 2>/dev/null); then
        echo "Error: Unable to get repository owner"
        echo "Tip: Make sure you're in a git repository and authenticated with GitHub"
        exit 1
    fi
fi

if ! REPO=$(gh repo view --json name -q .name 2>/dev/null); then
    # Fallback to jq method
    if ! REPO=$(gh repo view --json name 2>/dev/null | jq -r '.name' 2>/dev/null); then
        echo "Error: Unable to get repository name"
        exit 1
    fi
fi

if [ -n "$ARGUMENTS" ]; then
    PR_NUM="$ARGUMENTS"
    echo "Using specified PR #$PR_NUM"
elif ! PR_NUM=$(gh pr view --json number | jq -r '.number' 2>/dev/null); then
    echo "Error: Unable to get PR number. Are you on a PR branch?"
    exit 1
fi

# Create classification filters (more maintainable approach)
cat > comment_classifier.jq << 'EOF'
def classify_priority:
  if test("(MUST|must|required|critical|blocking|error|fail)"; "i") then "HIGH_PRIORITY"
  elif test("(should|suggest|recommend|consider|maybe|could)"; "i") then "MEDIUM_PRIORITY"
  elif test("(nit|minor|style|typo|optional)"; "i") then "LOW_PRIORITY"
  else "UNKNOWN" end;

def is_actionable:
  test("(fix|change|add|remove|update|modify|implement|create|delete)"; "i");

.[] | {
  id: .id,
  author: .author.login,
  body: .body,
  type: (.body | classify_priority),
  actionable: (.body | is_actionable)
}
EOF

# Classify comments by urgency and type (using external jq file)
gh api repos/$OWNER/$REPO/pulls/$PR_NUM/comments | jq -f comment_classifier.jq

# Clean up temporary file
rm comment_classifier.jq
```

### Step 3: Smart Resolution Tracking
```bash
# Set up variables first with error handling
# Try with -q flag first (more reliable)
if ! OWNER=$(gh repo view --json owner -q .owner.login 2>/dev/null); then
    # Fallback to jq method
    if ! OWNER=$(gh repo view --json owner 2>/dev/null | jq -r '.owner.login' 2>/dev/null); then
        echo "Error: Unable to get repository owner"
        echo "Tip: Make sure you're in a git repository and authenticated with GitHub"
        exit 1
    fi
fi

if ! REPO=$(gh repo view --json name -q .name 2>/dev/null); then
    # Fallback to jq method
    if ! REPO=$(gh repo view --json name 2>/dev/null | jq -r '.name' 2>/dev/null); then
        echo "Error: Unable to get repository name"
        exit 1
    fi
fi

if [ -n "$ARGUMENTS" ]; then
    PR_NUM="$ARGUMENTS"
    echo "Using specified PR #$PR_NUM"
elif ! PR_NUM=$(gh pr view --json number | jq -r '.number' 2>/dev/null); then
    echo "Error: Unable to get PR number. Are you on a PR branch?"
    exit 1
fi

# Create a resolution checklist based on comment content
gh api repos/$OWNER/$REPO/pulls/$PR_NUM/comments | jq -r '.[] | select(.body | test("(todo|TODO|FIXME|suggestion|issue|problem|fix|change|update|add|remove|modify)"; "i")) | "- [ ] Comment \(.id) by \(.author.login): \(.body | split("\n")[0] | .[0:100])..."'
```

### Step 4: Reply and Resolve Comments (Enhanced with GraphQL)

```bash
# Main function: Resolve comment using GraphQL (Recommended)
resolve_review_comment() {
    local COMMENT_ID="$1"
    local RESOLUTION_MESSAGE="$2"
    
    echo "🔧 Resolving comment $COMMENT_ID..."
    
    # Use GraphQL method first (most reliable)
    if reply_and_resolve_by_id "$COMMENT_ID" "$RESOLUTION_MESSAGE"; then
        echo "✅ Successfully resolved comment $COMMENT_ID using GraphQL"
        return 0
    fi
    
    # Fallback to REST API method
    echo "⚠️ GraphQL method failed, trying REST API fallback..."
    reply_to_comment_rest "$COMMENT_ID" "$RESOLUTION_MESSAGE"
}

# Resolve by searching comment content
resolve_by_content() {
    local SEARCH_TEXT="$1"
    local RESOLUTION_MESSAGE="$2"
    
    echo "🔍 Resolving comment containing: '$SEARCH_TEXT'"
    reply_and_resolve_by_text "$SEARCH_TEXT" "$RESOLUTION_MESSAGE"
}

# Batch resolve multiple comments by content patterns
resolve_multiple_by_content() {
    declare -A COMMENT_PATTERNS
    COMMENT_PATTERNS["set -e"]="문제를 해결했습니다. Python 스크립트 실행 전에 set +e로 일시적으로 엄격 모드를 해제하고, 실행 후 종료 코드를 수집한 다음 set -e를 다시 활성화하도록 수정했습니다."
    COMMENT_PATTERNS["하드코딩된 가상환경"]="하드코딩된 가상환경 경로를 파라미터화했습니다. Poetry 가상환경을 자동으로 탐지하고, VENV_PATH 환경변수로 외부에서 주입할 수도 있습니다."
    COMMENT_PATTERNS["check_dir"]="check_dir 함수를 활용하여 필수 디렉토리들을 검증하도록 추가했습니다. 데이터 디렉토리, pretrained_models 디렉토리, 스크립트 디렉토리의 존재를 확인합니다."
    
    for SEARCH_TEXT in "${!COMMENT_PATTERNS[@]}"; do
        echo "Processing comments containing: '$SEARCH_TEXT'"
        resolve_by_content "$SEARCH_TEXT" "${COMMENT_PATTERNS[$SEARCH_TEXT]}"
        sleep 2  # Rate limiting prevention
    done
}

# Batch resolve by specific comment IDs
resolve_multiple_by_ids() {
    declare -A COMMENTS
    COMMENTS["2236694352"]="문제를 해결했습니다. set -e 옵션을 적절히 관리하여 스크립트 안정성을 향상시켰습니다."
    COMMENTS["2236694357"]="하드코딩된 가상환경 경로를 파라미터화했습니다. 이제 다양한 환경에서 유연하게 사용 가능합니다."
    COMMENTS["2236694368"]="check_dir 함수를 활용하여 필수 디렉토리들을 사전 검증하도록 개선했습니다."
    
    for COMMENT_ID in "${!COMMENTS[@]}"; do
        resolve_review_comment "$COMMENT_ID" "${COMMENTS[$COMMENT_ID]}"
        sleep 2  # Rate limiting prevention
    done
}

# Get unresolved review threads
get_unresolved_threads() {
    echo "🔍 Finding unresolved review threads..."
    
    # Ensure OWNER and REPO are set
    if [[ -z "$OWNER" || -z "$REPO" ]]; then
        echo "⚠️ Setting OWNER and REPO manually..."
        OWNER=$(gh repo view --json owner -q .owner.login 2>/dev/null || echo "")
        REPO=$(gh repo view --json name -q .name 2>/dev/null || echo "")
        
        if [[ -z "$OWNER" || -z "$REPO" ]]; then
            echo "❌ Failed to detect repository. Please set OWNER and REPO variables."
            echo "You can manually set: OWNER='your-org' REPO='your-repo'"
            return 1
        fi
    fi
    
    echo "Using repository: $OWNER/$REPO, PR #$PR_NUM"
    
    local GRAPHQL_RESPONSE=$(gh api graphql -f query="
    {
      repository(owner: \"$OWNER\", name: \"$REPO\") {
        pullRequest(number: $PR_NUM) {
          reviewThreads(first: 100) {
            nodes {
              id
              isResolved
              comments(first: 5) {
                nodes {
                  id
                  body
                  author { login }
                  createdAt
                }
              }
            }
          }
        }
      }
    }" 2>/dev/null)
    
    if [ $? -eq 0 ] && [ -n "$GRAPHQL_RESPONSE" ]; then
        # Clean control characters and parse JSON safely
        echo "$GRAPHQL_RESPONSE" | tr -d '\000-\031' | jq '.data.repository.pullRequest.reviewThreads.nodes[] | select(.isResolved == false) | {id: .id, firstComment: .comments.nodes[0].body | split("\n")[0], commentId: .comments.nodes[0].id}' 2>/dev/null || echo "Failed to parse review threads"
    else
        echo "Failed to fetch review threads"
    fi
}

# Example usage:
# resolve_review_comment "2236694357" "Fixed: Parameterized virtual environment path"
# resolve_by_content "set -e" "Fixed error handling as requested"
# resolve_multiple_by_content
# resolve_multiple_by_ids
# get_unresolved_threads
```

## Parallel Processing for Comment Resolution

Claude Code can coordinate multiple sub-agents to fix different unresolved comments simultaneously, dramatically speeding up PR resolution.

### When to Use Parallel Sub-Agents

Analyze the unresolved comments and use parallel processing when:

- Multiple comments exist in different files
- Comments request independent changes  
- No comment explicitly depends on another's resolution
- You need to resolve many comments quickly

### Advanced Parallel Resolution Pattern

```bash
# 1. First, get comprehensive comment analysis
echo "=== UNRESOLVED PR COMMENTS ANALYSIS ==="
if [ -n "$ARGUMENTS" ]; then
    PR_NUM="$ARGUMENTS"
    echo "Using specified PR #$PR_NUM"
else
    PR_NUM=$(gh pr view --json number | jq -r '.number')
    echo "Detected PR #$PR_NUM from current branch"
fi
OWNER=$(gh repo view --json owner | jq -r '.owner.login')  
REPO=$(gh repo view --json name | jq -r '.name')

# 2. Get all actionable comments with location context (with error handling)
if COMMENTS=$(gh api repos/$OWNER/$REPO/pulls/$PR_NUM/comments 2>/dev/null); then
    echo "$COMMENTS" | jq '.[] | select(.body | test("(suggestion|todo|fix|change|update|add|remove|modify|issue)"; "i")) | {
      id: .id,
      file: .path,
      line: .line, 
      body: .body,
      author: .author.login,
      created_at: .created_at,
      action_required: true
    }' > unresolved_comments.json 2>/dev/null || echo "Failed to parse comments"
else
    echo "Failed to fetch comments from API"
    echo "[]" > unresolved_comments.json
fi

# 3. Group by file for parallel processing
jq 'group_by(.file) | map({file: .[0].file, comments: map({id, line, body, author})})' unresolved_comments.json

# 4. Create parallel work plan
echo "=== PARALLEL RESOLUTION PLAN ==="
jq -r '.[] | "File: \(.file)\nComments: \(.comments | length)\nActions: \(.comments | map(.body | split("\n")[0]) | join("; "))\n"' unresolved_comments.json
```

### Parallel Sub-Agent Spawning Pattern

```
You: These comments look independent. Let's resolve them in parallel.

Claude: Analyzing unresolved comments...
Found 8 actionable comments across 4 files:

**Parallel Resolution Plan:**
- Sub-Agent 1: `app/models/chat.rb` (2 comments)  
  - Remove undefined tools from available_tools
  - Fix feature flag check
- Sub-Agent 2: `app/tools/search_emails.rb` (3 comments)
  - Fix N+1 query with preloading
  - Remove TODO comment  
  - Add caching strategy
- Sub-Agent 3: `test/` files (2 comments)
  - Fix test hygiene issues
  - Update test patterns
- Sub-Agent 4: `app/tools/concerns/` (1 comment)
  - Consider EmailFormatter location

Spawning 4 parallel sub-agents...
[Each agent works independently on their assigned files]
```

## Quality Assurance & Verification

### Before Committing Changes
```bash
# 1. Verify all tests pass (check project-specific commands first)
if [ -f "CLAUDE.md" ]; then
    echo "Checking CLAUDE.md for project-specific test commands..."
    # Look for common test patterns in CLAUDE.md
    if grep -q "poetry run" CLAUDE.md; then
        echo "Running Poetry-based tests..."
        poetry run pytest || poetry run python -m pytest
    elif grep -q "npm test\|yarn test" CLAUDE.md; then
        echo "Running Node.js tests..."
        npm test || yarn test
    elif grep -q "bundle exec" CLAUDE.md; then
        echo "Running Ruby tests..."
        bundle exec standardrb && bundle exec rspec
    elif grep -q "python -m pytest\|pytest" CLAUDE.md; then
        echo "Running Python tests..."
        python -m pytest || pytest
    else
        echo "No recognized test commands in CLAUDE.md. Running common defaults..."
        # Try common test commands
        if [ -f "package.json" ]; then
            npm test || yarn test
        elif [ -f "pyproject.toml" ]; then
            poetry run pytest || python -m pytest
        elif [ -f "Gemfile" ]; then
            bundle exec rspec || bundle exec test
        else
            echo "Warning: Could not determine appropriate test command"
        fi
    fi
else
    echo "No CLAUDE.md found. Trying to detect project type..."
    # Auto-detect project type and run appropriate tests
    if [ -f "package.json" ]; then
        npm test
    elif [ -f "pyproject.toml" ]; then
        poetry run pytest
    elif [ -f "requirements.txt" ]; then
        python -m pytest
    elif [ -f "Gemfile" ]; then
        bundle exec rspec
    elif [ -f "go.mod" ]; then
        go test ./...
    else
        echo "Warning: Could not determine project type for testing"
    fi
fi

# 2. Check that all comments have been addressed
echo "=== VERIFICATION: Checking comment resolution ==="
gh api repos/$OWNER/$REPO/pulls/$PR_NUM/comments | jq '.[] | {id: .id, body: .body | split("\n")[0], status: "NEEDS_VERIFICATION"}'

# 3. Create resolution summary
echo "=== RESOLUTION SUMMARY ==="
git log --oneline -n 5  # Show recent commits
git diff --stat HEAD~1  # Show what changed
```

### Post-Resolution Validation
```bash
# Verify the PR is ready for merge
gh pr checks            # Check CI status
gh pr view --json mergeable | jq '.mergeable'  # Check merge status
```

## Custom Slash Commands

For repeated workflows, store these improved templates:

### Enhanced Sequential Resolution
`.claude/commands/resolve-pr-comments.md`:
```markdown
Please resolve all comments and issues in the current PR systematically:

**Phase 1: Discovery**
1. Use GraphQL to gather ALL review threads and comments
2. Identify unresolved threads and classify by priority
3. Create a TodoWrite plan for systematic resolution

**Phase 2: Implementation**  
4. Work through each comment systematically
5. Make code changes, run tests, verify fixes
6. Use GraphQL to reply to comments and resolve threads
7. Verify thread resolution status
8. Mark todos as completed when finished

**Phase 3: Verification**
9. Run full linting and test suite
10. Verify all review threads have been resolved
11. Commit with descriptive message
12. Provide comprehensive resolution summary

**Usage Examples:**
- Text-based resolution: `resolve_by_content "set -e" "Fixed error handling"`
- ID-based resolution: `resolve_review_comment "123456" "Fixed as requested"`
- Batch processing: `resolve_multiple_by_content`
- Check status: `get_unresolved_threads`

**Focus Areas:** $ARGUMENTS
```

### Parallel Resolution Command
`.claude/commands/parallel-resolve.md`:
```markdown
Analyze and resolve PR comments in parallel:

1. **Discovery**: Get all actionable comments with file grouping
2. **Planning**: Create parallel work assignments by file/area  
3. **Execution**: Spawn sub-agents for independent comment groups
4. **Coordination**: Monitor progress and handle dependencies
5. **Integration**: Merge all changes and verify compatibility
6. **Quality**: Run comprehensive tests and validation

Target: Resolve $ARGUMENTS comments in parallel
```

## Troubleshooting Common Issues

### Authentication Problems
```bash
# If you get authentication errors:
gh auth login
gh auth refresh

# Check current authentication status:
gh auth status
```

### API Rate Limiting
```bash
# Check current rate limit:
gh api rate_limit

# If rate limited, wait or use a different authentication token:
export GITHUB_TOKEN="your_token_here"
```

### Missing Dependencies
```bash
# Install GitHub CLI (macOS):
brew install gh

# Install GitHub CLI (Linux):
curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg | sudo gpg --dearmor -o /usr/share/keyrings/githubcli-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | sudo tee /etc/apt/sources.list.d/github-cli.list > /dev/null
sudo apt update
sudo apt install gh

# Install jq (macOS):
brew install jq

# Install jq (Linux):
sudo apt install jq
```

### Branch Context Issues
```bash
# If "Unable to get PR number" error:
# Make sure you're on a branch with an associated PR:
gh pr list --author "@me"
gh pr checkout <pr-number>

# Or specify PR explicitly:
gh pr view <pr-number> --comments
```

### Network Connectivity
```bash
# Test GitHub connectivity:
curl -s https://api.github.com/user

# If behind corporate firewall, configure proxy:
gh config set http_proxy http://proxy.company.com:8080
gh config set https_proxy https://proxy.company.com:8080
```

## Performance Tips

### Batch Operations
- Group multiple API calls together when possible
- Use GitHub's GraphQL API for complex queries requiring multiple REST calls
- Cache results locally for repeated operations

### Large Repositories
- Use pagination for repositories with many comments: `gh api --paginate`
- Filter results early to reduce data processing
- Consider using GitHub's search API for specific comment types

This enhanced approach provides robust, tested methods for finding and resolving ALL types of PR comments efficiently.