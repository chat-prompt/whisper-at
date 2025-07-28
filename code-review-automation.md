# Code Review Automation 설정 가이드

## Personal Access Token (PAT) 설정 방법

CodeRabbit과 같은 외부 봇이 코드 리뷰 자동화를 실행할 때 권한 문제를 해결하기 위해 Personal Access Token이 필요합니다.

### 1. Personal Access Token 생성

1. GitHub 계정의 Settings > Developer settings > Personal access tokens > Tokens (classic)로 이동
2. "Generate new token" 클릭
3. Token 이름: `CLAUDE_PAT` (또는 원하는 이름)
4. 필요한 권한 선택:
   - `repo` (전체 선택)
   - `workflow`
5. "Generate token" 클릭 후 토큰 복사 (한 번만 표시되므로 안전하게 보관)

### 2. Repository Secret 추가

1. 리포지토리의 Settings > Secrets and variables > Actions로 이동
2. "New repository secret" 클릭
3. Name: `CLAUDE_PAT`
4. Secret: 복사한 Personal Access Token 붙여넣기
5. "Add secret" 클릭

### 3. 동작 확인

이제 CodeRabbit이나 다른 봇이 PR 리뷰 코멘트를 남기면:
- 자동으로 코드 분석이 실행됩니다
- 필요시 코드가 수정되고 새로운 PR이 생성됩니다
- 원본 PR에 알림 코멘트가 추가됩니다

## 주의사항

- Personal Access Token은 절대 코드에 직접 포함하지 마세요
- Token의 권한은 필요한 최소한으로 설정하세요
- Token이 노출된 경우 즉시 revoke하고 새로 생성하세요

## 워크플로우 흐름

1. CodeRabbit 등의 봇이 PR에 리뷰 코멘트 작성
2. GitHub Actions가 코멘트 감지
3. Claude가 코멘트 분석 및 코드 수정
4. 수정사항이 있으면 새 브랜치와 PR 생성
5. 원본 PR 작성자가 확인 후 승인