import numpy as np
from mayavi import mlab

# 라이다 데이터 파일 경로
file_path = './vel/000011.bin'

# 라이다 데이터 읽기
data = np.fromfile(file_path, dtype=np.float32).reshape(-1, 4)

# x축이 0 이상인 데이터 포인트들만 필터링
filtered_data = data[data[:, 0] >= 0]

# 필터링된 데이터에서 x, y, z 좌표 추출
x = filtered_data[:, 0]
y = filtered_data[:, 1]
z = filtered_data[:, 2]

# Mayavi를 사용한 시각화
mlab.figure(bgcolor=(0, 0, 0))  # 배경색 설정
mlab.points3d(x, y, z, color=(0, 1, 0), mode='point')  # 포인트 클라우드 표시
mlab.axes()  # 축 표시
mlab.show()  # 시각화 창 표시
