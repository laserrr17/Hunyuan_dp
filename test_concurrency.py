import asyncio
import httpx
import time

# 配置
URL = "http://127.0.0.1:8013/generate_image"
CONCURRENCY = 3  # 并发请求数量
REQUESTS = 3 # 总请求数量

# 示例请求数据
payload = {
    "prompt": "制作一个GTA的宣传海报，人物是特朗普戴着墨镜",
    "output_path": "./output"
}

async def send_request(client):
    try:
        response = await client.post(URL, json=payload)
        response.raise_for_status()
        return response.json()
    except httpx.HTTPStatusError as e:
        print(f"Error response {e.response.status_code} while requesting {e.request.url!r}.")
    except Exception as e:
        print(f"An error occurred: {e}")

async def main():
    async with httpx.AsyncClient(timeout=None) as client:
        tasks = []
        start_time = time.time()

        # 创建并发任务
        for _ in range(REQUESTS):
            tasks.append(send_request(client))

        # 运行并发任务
        responses = await asyncio.gather(*tasks, return_exceptions=True)

        end_time = time.time()

        # 打印统计信息
        success_count = sum(1 for response in responses if isinstance(response, dict))
        error_count = REQUESTS - success_count
        total_time = end_time - start_time

        print(f"Total requests: {REQUESTS}")
        print(f"Successful responses: {success_count}")
        print(f"Failed responses: {error_count}")
        print(f"Total time taken: {total_time:.2f} seconds")
        print(f"Requests per second: {REQUESTS / total_time:.2f}")

if __name__ == "__main__":
    asyncio.run(main())