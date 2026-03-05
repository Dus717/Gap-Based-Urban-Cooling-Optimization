import csv
import time
import random
import requests
from bs4 import BeautifulSoup

def extract_cooling_centers_from_html(html_path):
    """
    从page_source.html中提取所有cooling center的名称和地址
    """
    with open(html_path, 'r', encoding='utf-8') as f:
        html_content = f.read()
    
    soup = BeautifulSoup(html_content, 'html.parser')
    cooling_centers = []
    
    # 查找所有包含cooling center信息的<li>元素
    li_elements = soup.find_all('li', class_='sc-jlZhew')
    
    for li in li_elements:
        # 查找名称
        name_element = li.find('a', class_='sc-dLMFU')
        if name_element:
            name = name_element.text.strip()
            
            # 查找地址
            address = ""
            # 获取<li>元素的所有文本内容
            li_text = li.get_text(separator='\n', strip=True)
            # 按换行符分割文本
            lines = li_text.split('\n')
            # 地址通常在第三行（索引为2）
            if len(lines) > 2:
                # 提取第三行作为地址
                potential_address = lines[2].strip()
                # 验证是否为有效的地址（包含数字和街道名称）
                if any(char.isdigit() for char in potential_address) and any(word in potential_address.upper() for word in ['AVENUE', 'STREET', 'ROAD', 'BLVD', 'DRIVE', 'LANE', 'PLACE', 'AVE', 'ST', 'RD']):
                    address = potential_address
            
            cooling_centers.append({'name': name, 'address': address})
    
    print(f"从HTML中提取到 {len(cooling_centers)} 个cooling center")
    return cooling_centers

def geocode_with_nominatim(address):
    """
    使用Nominatim API（OpenStreetMap）获取地址的经纬度
    """
    url = "https://nominatim.openstreetmap.org/search"
    headers = {'User-Agent': 'CoolingCentersBot/1.0'}
    # 在地址后面加上NYC
    params = {
        'q': f"{address}, NYC, New York City, NY",
        'format': 'json',
        'limit': 1
    }
    
    try:
        response = requests.get(url, params=params, headers=headers, timeout=30)
        if response.status_code == 200:
            data = response.json()
            if data:
                # 只返回第一个结果
                return float(data[0]['lat']), float(data[0]['lon'])
        else:
            print(f"  API请求失败: 状态码 {response.status_code}")
    except Exception as e:
        print(f"  地理编码失败: {e}")
    
    return None, None

def process_cooling_centers(cooling_centers, output_csv_path):
    """
    处理所有cooling center，获取经纬度并保存到CSV文件
    """
    try:
        # 准备CSV文件
        with open(output_csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['name', 'address', 'lat', 'lng'])
        
        # 处理每个cooling center
        total = len(cooling_centers)
        for i, center in enumerate(cooling_centers, 1):
            name = center['name']
            address = center['address']
            print(f"\n处理 {i}/{total}: {name}")
            if address:
                print(f"  地址: {address}")
            
            # 首先使用名称进行地理编码
            print("  使用名称进行编码...")
            lat, lng = geocode_with_nominatim(name)
            
            # 如果名称编码失败且有地址，则使用地址进行编码
            if lat is None and address:
                print("  名称编码失败，尝试使用地址...")
                lat, lng = geocode_with_nominatim(address)
            
            # 保存结果
            with open(output_csv_path, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([name, address, lat, lng])
            
            # 随机延迟，避免API限流
            delay = random.uniform(1, 3)
            print(f"  等待 {delay:.1f} 秒后继续...")
            time.sleep(delay)
        
        print(f"\n处理完成！结果已保存到 {output_csv_path}")
        
    except Exception as e:
        print(f"处理过程中发生错误: {e}")

if __name__ == "__main__":
    # 配置路径（使用绝对路径）
    html_path = "E:\\Benchmark-Dataset-For-Building\\data\\Googlereviews\\python_project\\google-maps-review-scraper\\page_source.html"
    output_csv_path = "E:\\Benchmark-Dataset-For-Building\\data\\Googlereviews\\python_project\\google-maps-review-scraper\\cooling_centers_location_final.csv"
    
    # 从HTML中提取cooling center
    cooling_centers = extract_cooling_centers_from_html(html_path)
    
    if cooling_centers:
        # 首先生成包含所有cooling center名称和地址的CSV文件
        print("\n生成cooling_centers_location.csv文件...")
        with open(output_csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['name', 'address', 'lat', 'lng'])
            for center in cooling_centers:
                writer.writerow([center['name'], center['address'], '', ''])
        print(f"✓ 已生成包含 {len(cooling_centers)} 个cooling center的CSV文件")
        
        # 尝试获取经纬度
        print("\n尝试获取经纬度...")
        process_cooling_centers(cooling_centers, output_csv_path)
        
        # 无论是否成功，都提示用户文件已生成
        print(f"\n✓ 任务完成！文件已保存到: {output_csv_path}")
        print("\n如果某些地点的经纬度获取失败，您可以：")
        print("1. 检查网络连接是否正常")
        print("2. 手动在Google Maps中搜索这些地点并记录经纬度")
        print("3. 尝试使用其他地理编码服务")
    else:
        print("未提取到cooling center，任务失败")
