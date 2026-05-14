import json
import os

# 1. Đọc file dataset gốc (Anh - Việt)
with open('D:/Code AIM Paper/datasets/opus100_envi/opus100_envi_test.json', 'r', encoding='utf-8') as f:
    old_data = json.load(f)

new_data = []
for item in old_data:
    new_item = item.copy()
    
    # 2. BẠN SỬA TÊN KEY Ở ĐÂY CHO KHỚP VỚI FILE GỐC NHÉ!
    # Ví dụ đổi chỗ 'en' và 'vi'
    new_item['source'] = item['target']
    new_item['target'] = item['source'] 
    
    new_data.append(new_item)

# 3. Lưu thành bộ dataset mới tên là phomt_vien
os.makedirs('D:/Code AIM Paper/datasets/opus100_vien', exist_ok=True)
with open('D:/Code AIM Paper/datasets/opus100_vien/opus100_vien_test.json', 'w', encoding='utf-8') as f:
    json.dump(new_data, f, ensure_ascii=False, indent=2)

print("Đã tạo xong bộ test lật ngược Vi-En!")