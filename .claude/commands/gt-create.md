## 커밋 메시지 작성 가이드

- 형식: **<type>(<scope>): <subject>**
- type: `feat | fix | docs | style | refactor | test | chore`
- subject: 50자 이하, 현재형 동사 + 한글
- 본문(선택): 왜(WHY)·무엇(WHAT)·영향(IMPACT)을 네-줄 이하로
1. 위의 커밋 메시지 작성 가이드에 따라 git staging에 있는 코드 변경 내역을 검사해서 커밋 메시지 작성
2. gt create -m "1번에서 작성한 커밋 메시지 내용" 으로 실행
3. gt ss 실행