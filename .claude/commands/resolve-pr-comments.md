# Claude Code PR 코멘트 해결 시스템 가이드

Claude Code를 사용하여 풀 리퀘스트의 모든 코멘트, TODO, 이슈를 체계적으로 해결하는 가이드입니다. Claude Code는 터미널에서 직접 작동하며, 컨텍스트를 이해하고 전체 프로젝트 구조를 파악하여 파일 편집 및 커밋 생성과 같은 실제 작업을 수행합니다.

## 📚 목차

1. [컨텍스트 인식](#1-컨텍스트-인식)
2. [워크플로우 접근법](#2-워크플로우-접근법)
3. [GitHub API 명령어](#3-github-api-명령어)
4. [코멘트 해결 워크플로우](#4-코멘트-해결-워크플로우)
5. [품질 보증 및 검증](#5-품질-보증-및-검증)
6. [성능 팁](#6-성능-팁)

## 1. 컨텍스트 인식

Claude Code는 두 가지 방식으로 PR 컨텍스트를 처리할 수 있습니다:

1. **자동 감지**: PR 브랜치에 있으면 Claude Code가 현재 브랜치와 연관된 PR 컨텍스트를 자동으로 감지합니다
2. **수동 지정**: `$ARGUMENTS`를 통해 특정 PR 번호를 지정할 수 있습니다

Claude Code의 동작:
- `$ARGUMENTS`에서 PR 번호가 제공되면 해당 번호를 사용
- 그렇지 않으면 현재 브랜치와 연관된 PR을 감지
- 모든 PR 코멘트와 리뷰 스레드를 자동으로 확인

## 2. 워크플로우 접근법

깊은 사고가 필요한 문제의 성능을 크게 향상시키는 연구/계획/구현 패턴을 따릅니다:

### 단계 1: 연구 및 분석

```
이 PR과 모든 코멘트를 종합적으로 분석하세요. 다음을 찾아보세요:
1. 해결되지 않은 모든 리뷰 코멘트와 대화
2. 코멘트에서 언급된 TODO 항목
3. 코드 리뷰에서 요청된 변경사항
4. 응답이 필요한 질문들

GitHub API를 체계적으로 사용하여 모든 코멘트 유형에 대한 완전한 데이터를 가져오세요.
```

### 단계 2: 계획 수립

```
분석을 바탕으로 해결되지 않은 모든 항목을 처리할 상세한 계획을 세우세요.
유형별로 그룹화하세요 (코드 변경, 문서화, 질문 응답).
중요도와 의존성을 기반으로 우선순위를 정하세요.
TodoWrite 도구를 사용하여 체계적으로 진행상황을 추적하세요.
```

### 단계 3: 구현

```
계획의 각 항목에 대한 솔루션을 구현하세요:
- 요청된 코드 변경사항 적용
- 필요에 따라 문서 업데이트
- 질문에 대한 응답 준비
- 모든 변경사항이 코드 품질을 유지하고 테스트를 통과하는지 확인
- 완료되면 각 todo를 완료로 표시
```

### 단계 4: 해결 및 검증

```
모든 항목을 처리한 후:
1. 린팅 및 테스트를 실행하여 모든 것이 작동하는지 확인
2. 수행된 모든 변경사항의 요약 작성
3. 명확한 메시지로 변경사항 커밋
4. 모든 리뷰 코멘트가 해결되었는지 확인
```

## 3. GitHub API 명령어

실제로 작동하는 올바른 API 엔드포인트를 사용하는 것이 핵심입니다. 다음은 테스트된 작동 명령어들입니다:

### 공통 헬퍼 함수

```bash
# 저장소 정보 설정 및 검증
setup_repo_variables() {
    echo "🔧 저장소 정보를 설정하는 중..."
    
    # GitHub CLI 설치 확인
    if ! command -v gh &> /dev/null; then
        echo "❌ 오류: GitHub CLI 'gh'가 설치되지 않았습니다"
        exit 1
    fi

    # jq 설치 확인
    if ! command -v jq &> /dev/null; then
        echo "❌ 오류: 'jq'가 설치되지 않았습니다"
        exit 1
    fi

    # GitHub 인증 확인
    if ! gh auth status &> /dev/null; then
        echo "❌ 오류: GitHub 인증이 되지 않았습니다. 'gh auth login'을 실행하세요"
        exit 1
    fi

    # API 레이트 리밋 확인
    RATE_LIMIT=$(gh api rate_limit | jq -r '.resources.core.remaining' 2>/dev/null)
    if [ "$RATE_LIMIT" -lt 10 ]; then
        echo "⚠️ 경고: GitHub API 레이트 리밋이 낮습니다 (남은 횟수: $RATE_LIMIT)"
    fi
    
    # 저장소 정보 가져오기
    if ! OWNER=$(gh repo view --json owner -q .owner.login 2>/dev/null); then
        if ! OWNER=$(gh repo view --json owner 2>/dev/null | jq -r '.owner.login' 2>/dev/null); then
            echo "❌ 오류: 저장소 소유자를 가져올 수 없습니다"
            echo "💡 팁: Git 저장소에 있고 GitHub에 인증되었는지 확인하세요"
            return 1
        fi
    fi

    if ! REPO=$(gh repo view --json name -q .name 2>/dev/null); then
        if ! REPO=$(gh repo view --json name 2>/dev/null | jq -r '.name' 2>/dev/null); then
            echo "❌ 오류: 저장소 이름을 가져올 수 없습니다"
            return 1
        fi
    fi

    echo "✅ 저장소 정보: $OWNER/$REPO"
    return 0
}

# PR 번호 설정
setup_pr_number() {
    if [ -n "$ARGUMENTS" ]; then
        PR_NUM="$ARGUMENTS"
        echo "📌 지정된 PR #$PR_NUM 사용"
    elif ! PR_NUM=$(gh pr view --json number | jq -r '.number' 2>/dev/null); then
        echo "❌ 오류: PR 번호를 가져올 수 없습니다. PR 브랜치에 있나요?"
        return 1
    else
        echo "🔍 현재 브랜치에서 PR #$PR_NUM 감지"
    fi
    return 0
}

# GraphQL 쿼리 실행 및 스레드 정보 가져오기
fetch_review_threads() {
    echo "📥 리뷰 스레드를 가져오는 중..."
    
    local GRAPHQL_RESPONSE=$(gh api graphql -f query="
    {
      repository(owner: \"$OWNER\", name: \"$REPO\") {
        pullRequest(number: $PR_NUM) {
          reviewThreads(first: 100) {
            nodes {
              id
              isResolved
              isOutdated
              comments(first: 10) {
                nodes {
                  id
                  body
                  author { login }
                  createdAt
                  path
                  line
                }
              }
            }
          }
        }
      }
    }" 2>/dev/null)
    
    if [ $? -eq 0 ] && [ -n "$GRAPHQL_RESPONSE" ]; then
        # 제어 문자 제거 및 JSON 파싱
        echo "$GRAPHQL_RESPONSE" | tr -d '\000-\031' | jq '.data.repository.pullRequest.reviewThreads.nodes' 2>/dev/null || echo "[]"
    else
        echo "❌ GraphQL 데이터 가져오기 실패"
        echo "[]"
    fi
}

# 안전한 GraphQL 응답 전송
safe_graphql_reply() {
    local THREAD_ID="$1"
    local REPLY_BODY="$2"
    
    echo "💬 스레드에 안전한 GraphQL 응답 추가 중: $THREAD_ID"
    
    # GraphQL 변이를 위한 임시 파일 생성
    local TEMP_MUTATION=$(mktemp)
    
    # 응답 본문을 JSON 문자열로 올바르게 이스케이프
    local ESCAPED_BODY=$(echo "$REPLY_BODY" | jq -Rs .)
    
    # GraphQL mutation을 직접 작성 (변수 사용)
    cat > "$TEMP_MUTATION" <<EOF
{
  "query": "mutation(\$threadId: ID!, \$body: String!) { addPullRequestReviewThreadReply(input: { pullRequestReviewThreadId: \$threadId, body: \$body }) { comment { id body } } }",
  "variables": {
    "threadId": "$THREAD_ID",
    "body": $ESCAPED_BODY
  }
}
EOF
    
    # 디버깅을 위한 JSON 미리보기
    echo "📋 생성된 JSON 미리보기:"
    head -c 200 "$TEMP_MUTATION"
    echo "..."
    
    local REPLY_RESULT=$(gh api graphql --input "$TEMP_MUTATION" 2>&1)
    rm -f "$TEMP_MUTATION"
    
    if echo "$REPLY_RESULT" | grep -q '"id"'; then
        echo "✅ 응답이 성공적으로 추가되었습니다"
        return 0
    else
        echo "❌ 응답 추가 실패: $REPLY_RESULT"
        echo "💡 팁: GraphQL API 권한을 확인하거나 PR 일반 코멘트 사용을 고려하세요"
        return 1
    fi
}
```

### 주요 API 작업

```bash
# 초기 설정 (모든 작업 전에 실행)
setup_repo_variables && setup_pr_number

# 1. 현재 PR 상세 정보 및 코멘트 보기
gh pr view $PR_NUM --comments

# 2. 모든 리뷰 코멘트 가져오기 (코드 라인별 코멘트)
gh api repos/$OWNER/$REPO/pulls/$PR_NUM/comments 2>/dev/null | \
    jq '.[] | {id: .id, author: .author.login, body: .body, created_at: .created_at, in_reply_to_id: .in_reply_to_id}' 2>/dev/null || \
    echo "❌ 코멘트 가져오기 실패"

# 3. 모든 리뷰 요약 가져오기 (승인/변경요청/코멘트)
gh api repos/$OWNER/$REPO/pulls/$PR_NUM/reviews | \
    jq '.[] | select(.state == "COMMENTED") | {id: .id, author: .author.login, body: .body, submitted_at: .submitted_at}' 2>/dev/null || \
    echo "❌ 리뷰 요약 가져오기 실패"

# 4. 일반 PR 코멘트 가져오기 (전체 PR 토론 코멘트)
gh pr view --json comments 2>/dev/null | \
    jq '.comments[] | {id: .id, author: .author.login, body: .body, created_at: .created_at}' 2>/dev/null || \
    echo "❌ PR 코멘트 가져오기 실패"
```

### 리뷰 코멘트 응답 함수

```bash
# 텍스트 내용으로 코멘트 찾아 응답하기
reply_by_text() {
    local SEARCH_TEXT="$1"
    local REPLY_BODY="$2"
    
    echo "🔍 다음 텍스트가 포함된 코멘트 검색 중: $SEARCH_TEXT"
    
    # 리뷰 스레드 가져오기
    local THREADS=$(fetch_review_threads)
    
    if [ "$THREADS" = "[]" ] || [ -z "$THREADS" ]; then
        echo "❌ 리뷰 스레드를 가져올 수 없습니다"
        return 1
    fi
    
    # 사용 가능한 코멘트 표시 (디버깅용)
    echo "📋 사용 가능한 코멘트:"
    echo "$THREADS" | jq -r '.[] | .comments.nodes[0].body | split("\n")[0:2] | join(" ")' 2>/dev/null | head -5
    
    # 검색 텍스트가 포함된 스레드 찾기
    local THREAD_ID=$(echo "$THREADS" | jq -r \
      --arg text "$SEARCH_TEXT" \
      '.[] | select(.comments.nodes[].body | test($text; "i")) | .id' 2>/dev/null | head -1)
    
    # 찾지 못한 경우 부분 일치 시도
    if [[ -z "$THREAD_ID" ]]; then
        for word in $SEARCH_TEXT; do
            if [ ${#word} -gt 3 ]; then  # 3글자보다 긴 단어만 시도
                THREAD_ID=$(echo "$THREADS" | jq -r \
                  --arg word "$word" \
                  '.[] | select(.comments.nodes[].body | test($word; "i")) | .id' 2>/dev/null | head -1)
                if [[ -n "$THREAD_ID" ]]; then
                    echo "🎯 단어로 스레드 찾음: $word"
                    break
                fi
            fi
        done
    fi
    
    if [[ -z "$THREAD_ID" ]]; then
        echo "❌ 다음 텍스트가 포함된 코멘트를 찾을 수 없습니다: $SEARCH_TEXT"
        return 1
    fi
    
    echo "🧵 스레드 ID 찾음: $THREAD_ID"
    
    # 안전한 GraphQL 응답 전송
    safe_graphql_reply "$THREAD_ID" "$REPLY_BODY"
}

# 코멘트 ID로 응답하기 (REST 및 GraphQL ID 모두 지원)
reply_by_id() {
    local COMMENT_ID="$1"
    local REPLY_BODY="$2"
    
    echo "🔍 코멘트 ID에 대한 스레드 찾는 중: $COMMENT_ID"
    
    # 리뷰 스레드 가져오기
    local THREADS=$(fetch_review_threads)
    
    if [ "$THREADS" = "[]" ] || [ -z "$THREADS" ]; then
        echo "❌ 리뷰 스레드를 가져올 수 없습니다"
        return 1
    fi
    
    local THREAD_ID=""
    
    # GraphQL ID 형식인지 확인 (PRRC_로 시작)
    if [[ "$COMMENT_ID" == PRRC_* ]]; then
        THREAD_ID=$(echo "$THREADS" | jq -r \
          --arg id "$COMMENT_ID" \
          '.[] | select(.comments.nodes[].id == $id) | .id' 2>/dev/null | head -1)
    else
        # 숫자 ID (REST API)인 경우 변환 필요
        echo "🔄 REST API ID를 GraphQL 형식으로 변환 중..."
        
        # REST API 코멘트를 가져와서 매핑 찾기
        local REST_COMMENTS=$(gh api repos/$OWNER/$REPO/pulls/$PR_NUM/comments 2>/dev/null)
        if [ $? -eq 0 ] && [ -n "$REST_COMMENTS" ]; then
            # 이 REST ID에 대한 GraphQL ID를 내용 매칭으로 찾기
            local COMMENT_BODY=$(echo "$REST_COMMENTS" | jq -r \
              --arg rest_id "$COMMENT_ID" \
              '.[] | select(.id == ($rest_id | tonumber)) | .body' 2>/dev/null)
            
            if [ -n "$COMMENT_BODY" ] && [ "$COMMENT_BODY" != "null" ]; then
                # 매칭되는 본문 내용을 가진 GraphQL 코멘트 찾기
                THREAD_ID=$(echo "$THREADS" | jq -r \
                  --arg body "$COMMENT_BODY" \
                  '.[] | select(.comments.nodes[].body == $body) | .id' 2>/dev/null | head -1)
            fi
        fi
    fi
    
    if [[ -z "$THREAD_ID" ]]; then
        echo "❌ 코멘트 ID에 대한 스레드를 찾을 수 없습니다: $COMMENT_ID"
        return 1
    fi
    
    echo "🧵 스레드 ID 찾음: $THREAD_ID"
    
    # 안전한 GraphQL 응답 전송
    safe_graphql_reply "$THREAD_ID" "$REPLY_BODY"
}

# 인덱스로 코멘트에 응답하기 (0, 1, 2...)
reply_by_index() {
    local COMMENT_INDEX="$1"
    local REPLY_BODY="$2"
    
    echo "🔧 인덱스로 코멘트에 응답 중: $COMMENT_INDEX"
    
    # 리뷰 스레드 가져오기
    local THREADS=$(fetch_review_threads)
    
    if [ "$THREADS" = "[]" ] || [ -z "$THREADS" ]; then
        echo "❌ 리뷰 스레드를 가져올 수 없습니다"
        return 1
    fi
    
    # 인덱스로 특정 스레드 가져오기
    local THREAD_ID=$(echo "$THREADS" | jq -r ".[$COMMENT_INDEX].id" 2>/dev/null)
    local COMMENT_PREVIEW=$(echo "$THREADS" | jq -r ".[$COMMENT_INDEX].comments.nodes[0].body | split(\"\n\")[0]" 2>/dev/null)
    
    if [[ -z "$THREAD_ID" || "$THREAD_ID" == "null" ]]; then
        echo "❌ 인덱스에서 스레드를 찾을 수 없습니다: $COMMENT_INDEX"
        echo "📊 사용 가능한 인덱스: $(echo "$THREADS" | jq '. | length')개 스레드"
        return 1
    fi
    
    echo "🧵 스레드 ID 찾음: $THREAD_ID"
    echo "📝 코멘트 미리보기: $COMMENT_PREVIEW"
    
    # 안전한 GraphQL 응답 전송
    safe_graphql_reply "$THREAD_ID" "$REPLY_BODY"
}

# REST API 폴백 방법 (레거시, 신뢰성 낮음)
reply_to_comment_rest() {
    local COMMENT_ID=$1
    local REPLY_MESSAGE=$2
    
    echo "🔄 코멘트 $COMMENT_ID에 REST API 응답 시도 중..."
    
    # 직접 응답 먼저 시도
    if gh api repos/$OWNER/$REPO/pulls/$PR_NUM/comments/$COMMENT_ID/replies \
        --method POST \
        -f body="$REPLY_MESSAGE" 2>/dev/null; then
        echo "✅ 직접 응답 성공"
    else
        echo "⚠️ 직접 응답 실패, 일반 PR 코멘트로 추가 중..."
        gh pr comment $PR_NUM --body "**코멘트 ID $COMMENT_ID에 대한 답변:**
$REPLY_MESSAGE"
        echo "✅ 일반 PR 코멘트로 추가됨"
    fi
}
```

### 해결되지 않은 코멘트 필터링 및 분석

```bash
# 해결되지 않은 이슈를 나타내는 코멘트 찾기
get_unresolved_comments() {
    echo "🔍 해결되지 않은 코멘트 분석 중..."
    
    gh api repos/$OWNER/$REPO/pulls/$PR_NUM/comments 2>/dev/null | \
        jq '.[] | select(.body | test("(todo|TODO|FIXME|제안|이슈|문제|수정|변경|업데이트|추가|제거|수정)"; "i")) | {id: .id, body: .body, author: .author.login}' || \
        echo "❌ 코멘트 분석 실패"
}

# 후속 응답이 없는 코멘트 확인
get_comments_without_replies() {
    echo "🔍 응답이 없는 코멘트 확인 중..."
    
    gh api repos/$OWNER/$REPO/pulls/$PR_NUM/comments 2>/dev/null | \
        jq 'group_by(.in_reply_to_id // .id) | map(select(length == 1 and .[0].in_reply_to_id == null)) | flatten | .[] | {id: .id, body: .body, needs_response: true}' || \
        echo "❌ 응답 분석 실패"
}
```

## 4. 코멘트 해결 워크플로우

### 1단계: 종합적인 코멘트 발견
```bash
# 종합적인 코멘트 발견 워크플로우
discover_all_comments() {
    echo "🔍 모든 코멘트 발견 중..."
    
    # 초기 설정
    setup_repo_variables && setup_pr_number || return 1
    
    echo "📊 PR #$PR_NUM ($OWNER/$REPO) 코멘트 처리 중"
    
    # GraphQL을 사용하여 모든 리뷰 스레드와 코멘트를 한 번에 가져오기
    echo "=== GraphQL 기반 코멘트 발견 ==="
    
    local THREADS=$(fetch_review_threads)
    
    if [ "$THREADS" != "[]" ] && [ -n "$THREADS" ]; then
        echo "📊 리뷰 스레드 요약:"
        echo "$THREADS" | jq -r '.[] | "스레드: \(.id) | 해결됨: \(.isResolved) | 코멘트: \(.comments.nodes[0].body | split("\n")[0] | .[0:80])..."' 2>/dev/null
        
        echo ""
        echo "📝 해결되지 않은 코멘트:"
        echo "$THREADS" | jq -r '.[] | select(.isResolved == false) | "ID: \(.comments.nodes[0].id) | 스레드: \(.id) | \(.comments.nodes[0].body | split("\n")[0] | .[0:100])..."' 2>/dev/null
    else
        echo "⚠️ GraphQL 데이터 가져오기 실패"
    fi
}
```

### 2단계: 지능형 코멘트 분류

```bash
# 지능형 코멘트 분류 시스템
classify_comments_by_priority() {
    echo "🏷️ 코멘트를 우선순위별로 분류 중..."
    
    # 분류 필터 생성 (유지보수성 향상)
    cat > comment_classifier.jq << 'EOF'
def classify_priority:
  if test("(필수|중요|중대|차단|오류|실패|MUST|must|required|critical|blocking|error|fail)"; "i") then "높음"
  elif test("(제안|권장|고려|~해야|should|suggest|recommend|consider|maybe|could)"; "i") then "보통"
  elif test("(사소|스타일|오타|선택|nit|minor|style|typo|optional)"; "i") then "낮음"
  else "알 수 없음" end;

def is_actionable:
  test("(수정|변경|추가|제거|업데이트|구현|생성|삭제|fix|change|add|remove|update|modify|implement|create|delete)"; "i");

.[] | {
  id: .id,
  author: .author.login,
  body: .body,
  priority: (.body | classify_priority),
  actionable: (.body | is_actionable)
}
EOF

    # 우선순위와 유형별로 코멘트 분류
    gh api repos/$OWNER/$REPO/pulls/$PR_NUM/comments 2>/dev/null | \
        jq -f comment_classifier.jq || echo "❌ 코멘트 분류 실패"

    # 임시 파일 정리
    rm -f comment_classifier.jq
}
```

### 3단계: 스마트 해결 추적

```bash
# 해결 체크리스트 생성 시스템
create_resolution_checklist() {
    echo "📋 해결 체크리스트 생성 중..."
    
    # 코멘트 내용을 기반으로 해결 체크리스트 생성
    gh api repos/$OWNER/$REPO/pulls/$PR_NUM/comments 2>/dev/null | \
        jq -r '.[] | select(.body | test("(todo|TODO|FIXME|제안|이슈|문제|수정|변경|업데이트|추가|제거|수정)"; "i")) | "- [ ] 코멘트 \(.id) (\(.author.login)): \(.body | split("\n")[0] | .[0:100])..."' || \
        echo "❌ 체크리스트 생성 실패"
}
```

### 4단계: 코멘트에 응답하기

```bash
# 메인 함수: GraphQL을 사용한 코멘트 응답 (권장)
reply_to_review_comment() {
    local COMMENT_ID="$1"
    local REPLY_MESSAGE="$2"
    
    echo "🔧 코멘트 $COMMENT_ID에 응답 중..."
    
    # GraphQL 방법 우선 사용 (가장 안정적)
    if reply_by_id "$COMMENT_ID" "$REPLY_MESSAGE"; then
        echo "✅ GraphQL을 사용하여 코멘트 $COMMENT_ID에 성공적으로 응답했습니다"
        return 0
    fi
    
    # REST API 방법으로 폴백
    echo "⚠️ GraphQL 방법 실패, REST API 폴백 시도 중..."
    reply_to_comment_rest "$COMMENT_ID" "$REPLY_MESSAGE"
}

# 코멘트 내용으로 검색하여 응답
reply_by_content() {
    local SEARCH_TEXT="$1"
    local REPLY_MESSAGE="$2"
    
    echo "🔍 '$SEARCH_TEXT'가 포함된 코멘트에 응답 중"
    reply_by_text "$SEARCH_TEXT" "$REPLY_MESSAGE"
}

# 일괄 응답 함수들
batch_reply_by_patterns() {
    declare -A COMMENT_PATTERNS
    # 사용 예시 - 실제 사용 시 패턴을 수정하세요
    COMMENT_PATTERNS["에러"]="❌ 지적해주신 문제를 해결했습니다."
    COMMENT_PATTERNS["개선"]="✅ 제안해주신 개선사항을 적용했습니다."
    
    for SEARCH_TEXT in "${!COMMENT_PATTERNS[@]}"; do
        echo "📝 '$SEARCH_TEXT' 패턴 코멘트 처리 중..."
        reply_by_content "$SEARCH_TEXT" "${COMMENT_PATTERNS[$SEARCH_TEXT]}"
        sleep 2  # 레이트 리밋 방지
    done
}

# 모든 해결되지 않은 코멘트에 순차적으로 응답
interactive_reply_to_all() {
    echo "🔧 해결되지 않은 모든 코멘트에 응답 중..."
    
    local THREADS=$(fetch_review_threads)
    
    if [ "$THREADS" = "[]" ] || [ -z "$THREADS" ]; then
        echo "❌ 리뷰 스레드를 가져올 수 없습니다"
        return 1
    fi
    
    # 해결되지 않은 스레드 처리
    local THREAD_COUNT=0
    echo "$THREADS" | jq -c '.[] | select(.isResolved == false)' 2>/dev/null | while read -r thread; do
        THREAD_COUNT=$((THREAD_COUNT + 1))
        local THREAD_ID=$(echo "$thread" | jq -r '.id')
        local FIRST_LINE=$(echo "$thread" | jq -r '.comments.nodes[0].body | split("\n")[0]')
        
        echo "📝 스레드 $THREAD_COUNT: $FIRST_LINE"
        echo "🧵 스레드 ID: $THREAD_ID"
        
        # 사용자에게 응답 요청
        echo "이 코멘트에 대한 응답을 입력하세요 (건너뛸 경우 'skip'):"
        read -r REPLY_BODY
        
        if [[ "$REPLY_BODY" != "skip" && -n "$REPLY_BODY" ]]; then
            if safe_graphql_reply "$THREAD_ID" "$REPLY_BODY"; then
                echo "✅ 응답이 성공적으로 추가되었습니다"
            else
                echo "❌ 응답 추가 실패"
            fi
        else
            echo "⏭️ 건너뜀"
        fi
        
        sleep 1
    done
}

```

## 5. 품질 보증 및 검증

```bash
# 변경사항 커밋 전 검증
verify_before_commit() {
    echo "🔍 커밋 전 검증 중..."
    
    # CLAUDE.md에서 프로젝트별 테스트 명령어 확인
    if [ -f "CLAUDE.md" ]; then
        echo "📋 CLAUDE.md에서 프로젝트별 테스트 명령어 확인 중..."
        if grep -q "poetry run" CLAUDE.md; then
            echo "🧪 Poetry 기반 테스트 실행 중..."
            poetry run pytest || poetry run python -m pytest
        elif grep -q "npm test" CLAUDE.md; then
            echo "🧪 Node.js 테스트 실행 중..."
            npm test
        else
            echo "⚠️ 인식된 테스트 명령어가 없습니다"
        fi
    fi
    
    # 모든 코멘트가 해결되었는지 확인
    echo "📝 코멘트 해결 상태 확인 중..."
    gh api repos/$OWNER/$REPO/pulls/$PR_NUM/comments 2>/dev/null | \
        jq '.[] | {id: .id, body: .body | split("\n")[0], status: "확인_필요"}' || \
        echo "❌ 코멘트 상태 확인 실패"
}
```

## 6. 성능 팁

- **일괄 작업**: 가능한 경우 여러 API 호출을 함께 그룹화
- **GraphQL 우선**: 복잡한 쿼리에는 REST API보다 GraphQL 사용
- **결과 캐싱**: 반복 작업을 위해 결과를 로컬에 캐시
- **페이지네이션**: 많은 코멘트가 있는 저장소에서는 `gh api --paginate` 사용

## 사용 예시

```bash
# 초기 설정
setup_repo_variables && setup_pr_number

# 모든 코멘트 발견
discover_all_comments

# 우선순위별 분류  
classify_comments_by_priority

# 응답 방법들
reply_by_text "에러" "문제를 해결했습니다"
reply_by_index 0 "첫 번째 코멘트에 대한 응답"
interactive_reply_to_all  # 대화형 응답

# 검증 및 커밋
verify_before_commit
```
