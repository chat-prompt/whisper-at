# gt create and submit

Graphite를 사용한 커밋 생성 및 제출 가이드

## 커밋 메시지 작성 가이드

- 형식: **<type>(<scope>): <subject>**
- type: `feat | fix | docs | style | refactor | test | chore`
- subject: 50자 이하, 현재형 동사 + 한글
- 본문(선택): 왜(WHY)·무엇(WHAT)·영향(IMPACT)을 네-줄 이하로

1. git staging에 있는 코드 변경 내역을 검사해서 위의 커밋 메시지 작성 가이드에 따라 커밋 메시지 작성
   1. 변경사항 확인:
    ```bash
    git status
    git diff --staged
    ```
2. gt create -m "2번에서 작성한 커밋 메시지 내용" 으로 실행
3. gt submit --stack 실행
