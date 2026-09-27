# physics-visual-lab
physics course visualization 

## semiconductor

### ge-crystal (HW01-2)
3d crystal structure 파트.
- Ge Diamond cubic unit cell을 볼 수 있으며, 원자 클릭하면 nearest-neighbor 4개 연결되고 거리 d_{NN} 표시됨.
- latice constant a를 바꾸면 d_{NN}=asqrt3/4, n_{Ge}=8/a^3 
```
ge-crystal 
├── app.py     # PySide6 창 + PyVista + YAML + slider를 연결
├── crystal.py      # diamond 구조 생성 + periodic image + nearest neighbor 탐색
├── properties.py   # 단위 변환과 물리량 계산
└── visualization.py    # PyVista 쪽만 담당

```
파이프라인 
```
[FCC Bravais lattice]
       R points
          │
          │ attach
          ▼
[Ge basis]
b1=(0,0,0)
b2=(¼,¼,¼)
          │
          │ r = R + b
          ▼
[Ge diamond crystal]
          │
          ├─ atom click
          │    ↓
          │  4 nearest neighbors
          │    ↓
          │  dNN = √3a/4
          │
          ├─ nGe = 8/a³
          ├─ mass density
          └─ valence electron density
```
#### notations 
| 기호 | 뜻 | 설명 |
|---|---|---|
| \(a\) | lattice constant = 정육면체 unit cell 한 변 길이 | \(5.66\,Å\) |
| \(d_{NN}\) | nearest-neighbor distance | 가장 가까운 Ge 원자 중심 사이 거리 |
| \(N_{\text{atom}}\) | unit cell **하나 안의 원자 수** | diamond cubic Ge → 8개 |
| \(V_{\text{cell}}\) | unit cell 부피 | \(a^3\) |
| \(n_{\text{Ge}}\) | **단위 부피당 Ge 원자 수** = number density | \(\#/cm^3\) |
| \(M_{\text{Ge}}\) | Ge 1 mol의 질량 | \(72.6\,g/mol\) |
| \(\rho\) | mass density | \(g/cm^3\) |    

### crystal_structure
목적: 
    diamond cubic crystal structure 구조 이해. 왜 diamond = FCC + (1/4,1/4,1/4)인지 시각화
구현:
    [1] FCC lattice
    [2] FCC의 tetrahedral sites 표시
    [3] (1/4, 1/4, 1/4) 위치 확인
    [4] 그 위치를 두 번째 basis atom으로 선택
    [5] 모든 FCC lattice point에 같은 basis 반복
    [6] Diamond cubic
    [7] 한 원자의 4 nearest neighbors 연결
        tetrahedral coordination 확인
    [8] 두 번째 basis atom의 displacement를 t로 만들어서 slider 
        -> t=1/4에서 diamond 됨을 보이기 
설명:
    Diamond cubic = FCC Bravis lattice + {(0,0,0),(0.25,0.25,0.25)} basis 
    FCC의 tetrhedral site가 0.25 위치에 있음. 
    {(0, 0, 0) + (0, 0.5, 0.5) + (0.5, 0, 0.5) + (0.5, 0.5, 0)} / 4 
    = (0.25, 0.25, 0.25) 임을 보이기.
