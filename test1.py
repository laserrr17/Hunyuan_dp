import httpx

# 定义API的URL
base_url = "http://127.0.0.1:8013"

# 生成初始图片的请求
def test_generate_image():
    prompt_request = {
        "prompt": "一个红苹果",
        "output_path": "./output_images"
    }
    response = httpx.post(f"{base_url}/generate_image", json=prompt_request, timeout=60.0)
    
    if response.status_code == 200:
        result = response.json()
        print("Initial image generated.")
        print(f"Image paths: {result['image_paths']}")
        print(f"Session ID: {result['session_id']}")
        return result['session_id'], result['image_paths']
    else:
        print(f"Failed to generate image: {response.text}")
        return None, None

# 调整图片的请求
def test_adjust_image(session_id, additional_prompt):
    adjust_request = {
        "session_id": session_id,
        "prompt": additional_prompt
    }
    response = httpx.post(f"{base_url}/adjust_image", json=adjust_request, timeout=60.0)
    
    if response.status_code == 200:
        result = response.json()
        print("Adjusted image generated.")
        print(f"Adjusted image paths: {result['image_paths']}")
        return result['image_paths']
    else:
        print(f"Failed to adjust image: {response.text}")
        return None

# 测试流程
def run_tests():
    session_id, image_paths = test_generate_image()
    if session_id:
        # 第一次调整图片
        adjusted_image_paths = test_adjust_image(session_id, "加一个苹果")
        # 第二次调整图片
        if adjusted_image_paths:
            test_adjust_image(session_id, "苹果变成绿色")
        # 第三次调整图片
        test_adjust_image(session_id, "加一个芒果")

if __name__ == "__main__":
    run_tests()