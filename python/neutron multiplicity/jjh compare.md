전지현씨 데이터 분석과의 비교
===

각각의 과정에 대해 전지현씨가 중간 결과 파일을 저장하였고, 이것을 내가 별도로 작성한 결과와 비교하여 차이점이 발생한다면, 그 차이점이 어떻게 발생하는지 확인한다.

## 1. Analysis

ptrac file (확장자 .p)를 읽고, 각각의 중성자, 감마선이 섬광검출이에 검출된 시간(time), 셀값(cell), history(NPS), origin 을 기록한다. 
전지현씨는 "C1_1_neutron_analysis.o"와 같은 파일명으로 저장하였으며, 이것을 내가 python으로 작성한 결과와 비교한 결과 동일하였다.
C3_1.p 서부터 C3_20.p 파일까지 모두 읽었으며 각각의 파일에서 추출한 데이터가 동일함을 확인하였다. 

## 2. Reduce

reduce는 같은 Ncell 과, NPS 를 가진 값을 최초의 값만 제외하고 제거한다. 아래의 것은 C3_2_neutron_analysis.o 의 결과의 일부이다. 19, 20, 23 번의 결과가 cell 값이 229, Nhist 값이 23으로 같지만 그 origin이 19, 20번은 탄소원자의 충돌이고, 23번은 양성자 생성반응으로 다르다. 
       

| Index |time |  cell |  Nhist | origin|
|:---:|---:|---:|---:|:---:|
|15 | 68971.0  | 126 |    18 |     P|
|16 | 21152.0  | 229 |    19 |     C|
|17 | 86553.0  | 221 |    21 |     C|
|18 |  5012.3  | 229 |    22 |     C|
|19 | 56672.0  | 229 |    23 |     C|
|20 | 56679.0  | 229 |    23 |     C|
|21 | 58438.0  | 328 |    23 |     C|
|22 | 56673.0  | 124 |    23 |     C|
|23 | 56670.0  | 229 |    23 |     P|
|24 | 14655.0  | 241 |    24 |     C|
|25 | 14655.0  | 241 |    24 |     C|

아래의 결과는 전지혜씨의 데이터 분석 결과인 "C3_2_neutron_reduced.o" 에서 앞의 analysis 결과에 의한 부분이다. 앞의 결과의 19, 20번이 아래 결과의 15번 으로 reduced 되었지만, 23번은 (18번에) 살아 있음을 알 수 있다. 

| Index |time |  cell |  Nhist | origin|
|:---:|---:|---:|---:|:---:|
|9  | 56555.0 |  342  |   14 | carbon scattering |
|10 | 86842.0 |  123  |   17 | carbon scattering|
|11 | 68971.0 |  126  |   18 | proton production|
|12 | 21152.0 |  229  |   19 | carbon scattering|
|13 | 86553.0 |  221  |   21 | carbon scattering|
|14 |  5012.3 |  229  |   22 | carbon scattering|
|15 | 56672.0 |  229  |   23 | carbon scattering|
|16 | 58438.0 |  328  |   23 | carbon scattering|
|17 | 56673.0 |  124  |   23 | carbon scattering|
|18 | 56670.0 |  229  |   23 | proton production|
|19 | 14655.0 |  241  |   24 | carbon scattering|

그러나 이렇게 reduced 시키는 원인을 살펴보면 하나의 검출기의 측정 시간 간격 안에 두개의 중성자 측정 반응이 일어날 경우 하나로 측정할 확률이 높기 때문이므로 같은 중성자 반응이라면 analysis 에서의 23번도 제거시키는게 맞다고 여겨진다. 따라서 아래와 같이 나의 데이터 처리는 analysis의 23번을 제거하는 것으로 하였다.

| Index |time |  cell |  Nhist | origin|
|:---:|---:|---:|---:|:---:|
|9  | 56555.0 |  342 |    14 |     C|
|10 | 86842.0 |  123 |    17 |     C|
|11 | 68971.0 |  126 |    18 |     P|
|12 | 21152.0 |  229 |    19 |     C|
|13 | 86553.0 |  221 |    21 |     C|
|14 |  5012.3 |  229 |    22 |     C|
|15 | 56672.0 |  229 |    23 |     C|
|16 | 58438.0 |  328 |    23 |     C|
|17 | 56673.0 |  124 |    23 |     C|
|18 | 14655.0 |  241 |    24 |     C|
|19 | 33643.0 |  325 |    26 |     P|


## 3. Combine & Order

Step 2 의 reduced 단계부터 전지혜씨의 분석과 나의 분석이 차이가 나서 combine & order 단계를 그대로 분석 할 수 없었다. 이에 

1.전지혜 씨의 reduced step 에서의 데이터를 읽어서
2.내가 작성한 combine & order 함수에서 처리하여
3.전지혜 시의 combine & order 결과와 비교하였 일치함을 확인하였다.


## 4. Rossi

전지혜 시의 코드는 아래와 같다.
```python
A = ['neutron','photon']

for x in range (1,25):
    for z in range (0,2):
        f_new1=open('C'+str(x)+'_'+A[z]+'.o',"r")
        f_new11=open('C'+str(x)+'_'+A[z]+'_rossi.o',"w")

        line1=f_new1.readlines()

            for i in range(0,len(line1)):
                for j in range(0,(len(line1)-i)):       
                    a = int(line1[i])
                    b = int(line1[i+j])
                    if (b-a) <= 100:
                        f_new11.write(str(b-a)+'\n')
                    elif (b-a) > 100:
                        break
        f_new1.close()
        f_new11.close()
```

여기서 ```b = int(line1[i+j])``` 을 보면 j=0 부터 시작하기때문에 b-a 는 자기 자신을 빼는 부분이 항상 들어가게 된다. 그런데 Rossi $\alpha$ 분포는 서로 다른 두 시간에 측정되는 중성자 혹은 광자의 시간 차이의 분포이므로 자기 자신을 빼는 것은 뭔가 이해 할 수 없다. 

