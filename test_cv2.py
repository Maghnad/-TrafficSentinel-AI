import cv2
import numpy as np

try:
    img = np.zeros((500, 500, 3), dtype=np.uint8)
    
    # Test 1: Literal pure types
    print("Test 1: literals")
    cv2.rectangle(img, (10, 10), (100, 100), (0, 0, 255), 2)
    
    # Test 2: Variables
    print("Test 2: variables")
    x1, y1, x2, y2 = 10, 10, 100, 100
    color = (0, 0, 255)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    
    # Test 3: Numpy types
    print("Test 3: numpy types")
    x1, y1, x2, y2 = np.int64(10), np.int64(10), np.int64(100), np.int64(100)
    color = (np.int64(0), np.int64(0), np.int64(255))
    cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (int(color[0]), int(color[1]), int(color[2])), 2)

    # Test 4: Recreating the exact loop scenario
    print("Test 4: exact scenario")
    bbox = [10.5, 20.2, 100.8, 200.9]
    has_violation = True
    x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
    color = (0, 0, 255) if has_violation else (0, 255, 0)
    color_tuple = (int(color[0]), int(color[1]), int(color[2]))
    if has_violation:
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color_tuple, 2)
    else:
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color_tuple, 1)
        
    print("All tests passed!")
except Exception as e:
    import traceback
    print("ERROR:", e)
    traceback.print_exc()
