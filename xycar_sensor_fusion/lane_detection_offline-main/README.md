# 차선인식 경진대회
본 프로젝트를 진행하기 전까지
- OpenCV 기초와 응용
- 허프변환 기반 차선 인식 주행
- 슬라이딩 윈도우 기반 차선 인식 주행
- 필터 기반 조향각 제어
- PID 기반 조향각 제어

위 내용으로 학습을 진행하며 차선 이탈 없이 자율 주행하기 위해 모든 조원들이 머리를 맞대고 열심히 실습을 진행했습니다. 위의 학습한 내용을 바탕으로 차선인식 경진대회에서 자이카가 차선을 이탈하지 않고 주행할 수 있도록 주어진 템플릿을 활용해서 프로젝트를 진행하시면 됩니다. 주어진 템플릿에 대한 간략한 설명을 읽고 템플릿을 수정하면서 원하는 주행을 할 수 있는 코드를 작성해주세요.

# 템플릿 설명
템플릿의 이름을 LaneKeepingSystem으로 변경한 다음 자이카의 xycar_ws의 src에 업로드하여 catkin_make를 하면 프로젝트의 템플릿으로 활용할 수 있습니다.
## CMakeLists.txt
프로젝트 코드를 Build하는 환경을 설정합니다. 프로젝트를 수행하기 위해서 필요한 패키지, 헤더 파일, 라이브러리, 파일 이름 등을 명시할 수 있으며 필요한 경우 ROS에서 지원하는 msg들을 추가하여 노드 간의 통신 형태를 추가할 수 있습니다.
![277145191-c17aac73-b4ea-48ff-8552-642f18759745](https://github.com/prgrms-ad-devcourse/notice-manage/assets/109266664/b2e3a7ab-4b22-4034-bf1f-cdbca5f604bc)<br>
기본 Libray
- LaneDetector
- LaneKeepingSystem
- MovingAverageFilter
- PIDController

기본 Package
- OpenCV
- YAML_CPP
- Catkin
- roscpp
- xycar_msgs
- sensor_msgs
- std_msgs

기본으로 탑재된 Library와 Package만으로도 프로젝트를 수행하는데 무리가 없으나 필요한 경우 원하는 패키지 및 라이브러리를 추가하여 사용할 수 있습니다.

## config.yaml
프로젝트를 진행하면서 프로젝트에서 사용하는 변수를 명시해 놓은 파일입니다. 파일에 명시되어 있는 Parameter 값을 수정하면서 각 팀의 프로젝트의 방향성에 알맞는 변수를 확인하여 수정하면서 프로젝트를 진행하면 됩니다. 

## drive.launch
여러분의 프로젝트에 필요한 센서를 동작시켜 원하는 Paramete값을 모두 받아올 수 있도록 해야합니다. 여러분의 차선인식주행에 필요한 데이터를 받아와서 활용할 수 있도록 launch 파일을 수정해서 사용해주세요.

## MovingAverageFilter
Moving Average Filter를 C++로 구현한 코드가 작성되어 있습니다. 해당 파일을 통해서 입력으로 들어오는 값에서 노이즈의 영향을 줄일 수 있습니다. 프로젝트 진행 과정에서 추가적으로 코드를 수정할 필요는 없으나 다른 형태의 필터를 구현해서 사용하고자 하는 경우 수정하여 사용할 수 있습니다. 
## PIDController
강의를 통해서 배운 PID 제어를 C++로 구현한 코드가 작성되어 있습니다. 자이카의 움직임을 제어하기 위해서 사용하는 코드로 PID를 통한 Feedback 제어를 하지 않을 경우 Xycar가 심하게 진동하면서 주행하는 것을 확인할 수 있습니다. PID는 강의에 나온 제어 기법으로 가장 간단하고 편리하게 사용할 수 있는 제어 방법입니다. 프로젝트 진행과정에서 PID 제어를 사용하지 않고자 한다면 다른 제어기법을 구현해서 프로젝트를 진행하면 됩니다.
![277145810-5078a7e7-650d-4fea-a87e-ad540e31167b](https://github.com/prgrms-ad-devcourse/notice-manage/assets/109266664/453c4c84-7356-4e87-b4a1-7409d3374da3)<br>
만약 PIDController을 그대로 사용하고자 한다면 config.yaml 파일에 명시되어 있는 PID 제어 값을 수정하면서 프로젝트를 진행하면 됩니다.

## LaneDetector
각자의 팀만의 고유한 영상처리 알고리즘을 구현하여 LaneKeepingSystem을 동작시킬 수 있는 코드를 작성해야 합니다. 여러분만의 영상처리 기법을 통해서 차선인식주행을 위해 필요한 영상처리 코드를 마음껏 작성해주세요.
### setConfiguration
![277146495-14b5e5d9-879e-418e-8848-93c00a47bffb](https://github.com/prgrms-ad-devcourse/notice-manage/assets/109266664/38516aab-385f-4eb1-bdc5-6bf56c85b278)<br>
영상 처리 알고리즘을 작성하기 위해서 필요한 일부의 변수들을 config.yaml 파일로부터 가져오는 함수입니다. 여러분의 영상처리 알고리즘을 작성하고자 할 때 추가적으로 필요한 변수가 있다면 변수를 config.yaml 파일에 형식에 맞춰 작성한 다음 해당 변수를 불러오는 함수를 추가해주세요.

### yourOwnFunction
![277146585-450fcbad-e4b4-4118-a246-1b1dc6c0aa4c](https://github.com/prgrms-ad-devcourse/notice-manage/assets/109266664/f14aeb5d-23e1-4f1a-87bf-de8324ecca38)<br>
여러분들만의 함수를 작성해서 원하는 형태의 영상처리 알고리즘을 구현해주세요. 차선인식 주행을 할 수 있다면 형태, 방식, 조건 모두 상관없습니다.

## LaneKeepingSystem
자이카의 차선인식 주행을 위해서 Main 함수에서 동작하는 run 함수가 포함되어 있는 파일입니다. 프로젝트에 필요한 모든 함수를 작성한 다음 해당 함수들을 이용해서 차선인식 주행을 할 수 있도록 코드를 작성해주세요.

### LaneKeepingSystem
![277145899-ecd9291a-73ed-441a-8e30-ab9e0be64ba1](https://github.com/prgrms-ad-devcourse/notice-manage/assets/109266664/92e3a510-3e64-471f-83c9-3b841729afc9)<br>
LaneKeepingSystem을 선언했을 때 동작하는 함수로 프로젝트를 initialize합니다. 작성한 LaneDetector 함수와 함께 동작하는 코드를 작성해주세요.

### setParams
![277145976-54c90421-fa71-472d-8c4e-c28c26221a82](https://github.com/prgrms-ad-devcourse/notice-manage/assets/109266664/c051d619-b3f5-4b79-92da-851dbe798908)<br>
LaneKeepingSystem이 동작하면서 필요한 변수들이 저장되어 있는 config.yaml 파일로부터 변수를 가져오는 함수입니다. 기본 템플릿에서 추가된 변수가 있는 경우 변수를 가져올 수 있도록 수정해주세요.

### ~LaneKeepingSystem
![277146064-f24ac2da-db62-426e-955c-f8bf9478e90f](https://github.com/prgrms-ad-devcourse/notice-manage/assets/109266664/abe495c8-0f4d-4ea9-bb0b-f912b741ebbe)<br>
LaneKeepingSystem이 동작이 끝나서 해당 함수를 종료하고자 하는 경우 선언한 Class들을 삭제하는 함수로 LaneDetector를 initialize한 경우 종료할 때 같이 삭제하는 코드를 작성해주세요.

### run
![277146104-6cf68a27-ca3a-44a7-ab11-4e9f21da64a1](https://github.com/prgrms-ad-devcourse/notice-manage/assets/109266664/4813d46a-0de9-4ee6-92ad-3c7fd2005fe2)<br>
Main 함수에서 동작할 run 코드를 작성해야 합니다. 작성한 모든 함수를 활용해서 차선인식 주행을 할 수 있도록 인지, 판단, 제어를 할 수 있는 run 함수를 작성해주세요.

### imageCallback
![277146151-8387261e-86b7-414e-9796-5f6cee342505](https://github.com/prgrms-ad-devcourse/notice-manage/assets/109266664/6eab92b3-6608-4a6a-8921-65342574680e)<br>
카메라로부터 들어온 이미지를 원하는 크기의 이미지로 return하는 함수입니다.

### speedControl
![277146239-82d5a89b-5856-45a6-bf19-0fb3c20924f8](https://github.com/prgrms-ad-devcourse/notice-manage/assets/109266664/26eaf5ba-9af8-44ed-912c-7b8867777983)<br>
자이카의 모터제어를 위해서 필요한 속도값을 들어오는 값에 따라서 속도를 감가속하는 함수입니다. 현재 코드에서는 정해진 차선의 기울기의 임계값에 따라서 큰 기울기를 가지고 있다면 감속하고 작은 기울기를 가지고 있다면 가속하는 형태로 되어 있습니다. 여러분의 프로젝트의 방향성에 맞춰서 해당 코드를 수정하거나 그대로 사용하시면 됩니다.

## drive
![277146340-c19e0421-0113-41c5-8873-2228c8cd0396](https://github.com/prgrms-ad-devcourse/notice-manage/assets/109266664/cbab1e18-8b4f-46d4-aa42-5cfeac4d4b7e)
<br>
자이카의 모터제어를 위해서 메시지를 발행하는 함수입니다. 모든 데이터 전처리 이후 원하는 모터의 조향각과 속도를 하나의 토픽으로 발행하여 발행된 값에 따라서 모터가 동작합니다.

## Main Code
여러분이 작성한 코드가 Main함수에서 동작합니다. 만약 여러분의 코드가 문제없이 작성되었다면 여러분의 프로젝트는 무리없이 빌드되고 실행될 것입니다.
