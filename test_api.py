import httpx
import pytest

base_url = "http://localhost:8013"  # Update this URL if your FastAPI server runs on a different address or port

@pytest.fixture(scope="module")
def http_client():
    with httpx.Client(base_url=base_url) as client:
        yield client

def test_generate_image(http_client):
    response = http_client.post("/generate_image", json={
        "prompt": "画一只小猫在阳光下玩耍",
        "output_path": "./output_images"
    })
    assert response.status_code == 200
    data = response.json()
    assert "session_id" in data
    assert "image_paths" in data
    assert len(data["image_paths"]) > 0
    print(f"Generated image paths: {data['image_paths']}")

def test_adjust_image(http_client):
    # First, generate an initial image
    generate_response = http_client.post("/generate_image", json={
        "prompt": "画一只小猫在阳光下玩耍",
        "output_path": "./output_images"
    })
    assert generate_response.status_code == 200
    generate_data = generate_response.json()
    session_id = generate_data["session_id"]

    # Then, adjust the image with a new prompt
    adjust_response = http_client.post("/adjust_image", json={
        "session_id": session_id,
        "prompt": "加上一个蓝色的蝴蝶"
    })
    assert adjust_response.status_code == 200
    adjust_data = adjust_response.json()
    assert "image_paths" in adjust_data
    assert len(adjust_data["image_paths"]) > 0
    print(f"Adjusted image paths: {adjust_data['image_paths']}")

def test_invalid_session(http_client):
    # Test adjusting an image with an invalid session ID
    adjust_response = http_client.post("/adjust_image", json={
        "session_id": "invalid_id",
        "prompt": "加上一个蓝色的蝴蝶"
    })
    assert adjust_response.status_code == 404
    assert adjust_response.json()["detail"] == "Session not found"

if __name__ == "__main__":
    pytest.main()