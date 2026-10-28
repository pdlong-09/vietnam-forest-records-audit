# Hướng dẫn giải đoán sử dụng đất trên ảnh vệ tinh

**Bản nháp 1, ngày 21/10/2026.**

Tài liệu dành cho giải đoán. Hướng dẫn dựa trên giao thức của JRC (Colditz et al. 2025, *JRC141231*, mục 2.3) và bộ lớp sử dụng đất của FAO FRA 2020 Remote Sensing Survey (FAO 2022, mục 3.1). Những chỗ chúng tôi bổ sung được ghi là **[bổ sung]**.

---

## 1. Việc anh/chị làm

Với mỗi điểm trên Collect Earth Online, anh/chị xem ảnh vệ tinh rồi trả lời hai câu:

1. Ngày **31/12/2020**, điểm này **có phải rừng không**? Nếu không phải rừng thì là loại đất gì?
2. Theo ảnh **mới nhất** có được, điểm này **đang được dùng vào việc gì**?

Trong lúc giải đoán, **không mở** bất kỳ bản đồ rừng hay bản đồ hiện trạng nào: bản đồ của nhà nước, GFC2020, Global Forest Watch, v.v.

## 2. Đơn vị giải đoán

- Trên màn hình có **ô vuông 100 × 100 m (1 ha)**, và ở giữa có **ô nhỏ 10 × 10 m**.
- **Nhãn là của ô nhỏ ở giữa.**
- Nhưng các tiêu chí diện tích, mật độ và mục đích sử dụng phải **xét trên cả ô 1 ha và vùng xung quanh** (JRC mục 2.3.2). Ví dụ:
  - Ô nhỏ rơi vào một khoảng trống nhỏ (dưới 0,5 ha) giữa rừng → vẫn là **rừng** (JRC Hình 8B).
  - Ô nhỏ rơi vào một thửa nương rẫy rộng trên 0,5 ha nằm sát rừng → **không rừng**, kể cả khi phần lớn ô 1 ha là rừng (JRC Hình 8E).

## 3. Định nghĩa rừng (EUDR Điều 2(4), FAO FRA 2020)

Rừng là đất thoả **đồng thời**:
- **diện tích** trên 0,5 ha;
- **cây cao** trên 5 m, hoặc cây có khả năng đạt 5 m tại chỗ;
- **độ che phủ tán** trên 10%;
- **không** phải đất chủ yếu dùng cho **nông nghiệp** hay **đô thị**.

Lưu ý khi áp dụng định nghĩa:
- **Mục đích sử dụng quyết định.** Vườn cây nông nghiệp (cà phê, cao su, điều, tiêu, cây ăn quả, cà phê trồng dưới tán cây che bóng) **không phải rừng**, dù cây cao, dày và rộng (JRC Hình 8F; EUDR Điều 2(5)–(6)).
- **Đất rừng tạm thời không có cây** sau khai thác, cháy hay gió đổ, nếu sẽ tái sinh hoặc trồng lại, **vẫn là rừng**. Chọn "Rừng sẽ tái sinh" ở F5.
- **Rừng mới trồng** có cây sẽ đạt ngưỡng khi trưởng thành **vẫn là rừng**.
- **Nương rẫy bỏ hoá** có cây đang tái sinh, đã hoặc sẽ đạt ngưỡng, **là rừng**. Nếu đang canh tác thì **không phải rừng**.
- **Tre nứa** và **cây họ cau** tính là rừng nếu đạt các ngưỡng và mục đích sử dụng là lâm nghiệp.
- Hệ **Taungya**, tức trồng cây nông nghiệp xen chỉ trong vài năm đầu của chu kỳ rừng trồng, **là rừng**.
- **Không cần đếm cây.** Chỉ ước lượng nhanh độ che phủ có trên 10% không; vài cây lớn thường đã đủ (JRC mục 2.3.3).

Chiều cao là tiêu chí khó nhất. Có thể ước lượng bằng:
- bóng cây, có tính đến vĩ độ và mùa chụp;
- ảnh Google Street View ở gần, nếu có;
- trình tự ảnh qua các năm.

### 3b. Khi ảnh không đủ để quyết định [bổ sung]

Không có dữ liệu thực địa. Đây cũng là điều kiện khi JRC giải đoán 21.752 điểm và FAO giải đoán hơn 400.000 điểm. Nhãn ở đây là **kết quả giải đoán ảnh của chuyên gia, không phải số đo thực địa**. Vì vậy, khi ảnh không đủ để quyết định, anh/chị **không được đoán cho có**, mà ghi lại mức độ không chắc chắn.

| Tiêu chí | Ảnh xác định được đến đâu | Khi chưa chắc |
|---|---|---|
| Diện tích trên 0,5 ha | Đo được trên ảnh độ phân giải cao: tốt | Ghi F6 theo khoảng gần nhất |
| Độ che phủ trên 10% | Ước lượng được trên ảnh độ phân giải cao: tốt | Độ tin cậy thấp, chọn F4b "độ che phủ" |
| Cây cao trên 5 m | **Khó.** Dựa vào bóng cây, kích thước tán, Street View và **tuổi cây theo chuỗi ảnh** (năm trồng nhìn thấy được). Thường chỉ khó với rừng trồng non, cây bụi và đất bỏ hoá đang tái sinh | Độ tin cậy thấp, chọn F4b "chiều cao". Nếu mục đích sử dụng rõ là lâm nghiệp và loài cây sẽ cao trên 5 m, áp dụng vế "có khả năng đạt ngưỡng": vẫn là rừng |
| Mục đích sử dụng (nông nghiệp hay lâm nghiệp) | Dựa vào hàng lối, khoảng cách cây, kết cấu tán, đường lô, nhà, chuỗi ảnh. Khó nhất là **cà phê dưới tán cây che bóng** và vườn hỗn hợp | Độ tin cậy thấp, chọn F4b "mục đích sử dụng"; ghi lớp nghiêng về nhiều hơn |
| "Sẽ tái sinh" (đất rừng tạm thời không có cây) | Có lợi thế nhìn lại: đã có ảnh 2021–2026, nên **thấy được** điểm này thực tế có tái sinh hay được trồng lại không | Theo JRC (mục 2.3.6): nếu bị phá **trước** năm 2020 và đã có dấu hiệu chuyển sang mục đích khác → không rừng. Nếu bị phá gần mốc 31/12/2020 → độ tin cậy thấp, chọn F4b "có thay đổi gần năm 2020", ghi năm bị phá ở F9 |

**Quy tắc chung:**
1. Chọn nhãn có khả năng đúng cao hơn, **đánh độ tin cậy thấp và ghi lý do**.
2. Chỉ chọn "Không xác định được" ở F1 khi thật sự không có ảnh dùng được (F5 = không có ảnh hoặc chỉ có ảnh độ phân giải thấp).
3. Không mở bản đồ chiều cao tán, bản đồ rừng hay lớp GEDI để quyết định. Những lớp này có chung nguồn dữ liệu với các bản đồ đang được đánh giá. Nhóm nghiên cứu sẽ dùng chúng riêng để kiểm tra sau.

## 4. Ảnh nên dùng

- Ảnh độ phân giải cao gần ngày 31/12/2020: lịch sử ảnh của Google Earth Pro, Esri Wayback, các lớp ảnh trong Collect Earth Online.
- Ảnh ghép Planet NICFI hằng tháng, từ năm 2020 trở đi.
- Sentinel-2 để xem diễn biến.
- Nên xem **chuỗi ảnh qua nhiều năm**, kể cả ảnh **sau** năm 2020: rừng lớn chậm nhưng có thể mất rất nhanh. Ảnh năm 2024 còn rừng là bằng chứng chắc rằng năm 2020 đã có rừng; ảnh năm 2019 có rừng thì chưa chắc (JRC mục 4.2).
- Ghi lại nguồn ảnh và ngày ảnh đã dùng (F8, F7b).

## 5. Phiếu giải đoán

| Trường | Câu hỏi | Lựa chọn |
|---|---|---|
| **F1 Rừng 2020** (JRC) | Ô giữa có phải rừng ngày 31/12/2020 không? | Rừng · Không rừng · Không xác định được |
| **F2 Loại 2020** (JRC) | Nếu **rừng** | Rừng nguyên sinh hoặc tái sinh tự nhiên · Rừng trồng |
| | Nếu **không rừng** | Không có cây hoặc cây bụi · Đất có cây gỗ khác (cây bụi, cây thấp che phủ trên 10%) · Cây phục vụ nông nghiệp · Cây trong đô thị · Cây ngoài rừng (ví dụ cây rải rác trên bãi chăn thả) · Đất không rừng nằm trong vùng rừng (mảng không rừng trên 0,5 ha bên trong vùng rừng) |
| **F3 Sử dụng đất chi tiết 2020** [bổ sung] | Ô giữa là gì? | Xem danh sách lớp ở mục 6 |
| **F4 Độ tin cậy** (JRC) | Độ tin cậy với câu F1 | Cao · Thấp |
| **F4b Lý do khi độ tin cậy thấp** [bổ sung, theo đề xuất của JRC mục 4.2] | Vì sao chưa chắc? | Gần ranh giới hoặc ảnh lệch vị trí · Khó xác định chiều cao · Khó xác định độ che phủ · Khó xác định mục đích sử dụng · Có thay đổi gần năm 2020 · Khác |
| **F5 Vấn đề** (JRC) | Vấn đề chính khi giải đoán | Không có · Mây · Không có ảnh gần năm 2020 · Chỉ có ảnh độ phân giải thấp · Rừng sẽ tái sinh · Nhiều loại sử dụng đất · Đất thưa cây (rừng thưa hoặc vườn thưa) · Khác (ghi ở phần ghi chú) |
| **F6 Kích thước mảng** [bổ sung] | Mảng đồng nhất chứa ô giữa rộng khoảng bao nhiêu? | Dưới 0,3 ha · Từ 0,3 đến 0,5 ha · Trên 0,5 ha |
| **F7 Sử dụng đất hiện tại** [bổ sung] | Theo ảnh mới nhất, ô giữa là gì? | Như danh sách lớp F3 |
| **F7b Ngày ảnh mới nhất** | | tháng/năm và nguồn ảnh |
| **F8 Ảnh dùng cho năm 2020** | | nguồn ảnh và ngày ảnh (có thể nhiều ảnh) |
| **F9 Ghi chú** | Chỉ ghi khi có điều đặc biệt | chữ tự do |

## 6. Danh sách lớp cho F3 và F7

**Rừng**
- R1 Rừng tự nhiên, kể cả rừng thứ sinh và rừng khộp
- R2 Rừng tre nứa
- R3 Rừng trồng lấy gỗ (keo, bạch đàn, thông, …)
- R4 Rừng mới trồng, chưa khép tán
- R5 Đất rừng tạm thời không có cây (sau khai thác hoặc cháy, sẽ tái sinh)
- R6 Nương rẫy bỏ hoá có cây đang tái sinh

**Cây nông nghiệp lâu năm**
- N1 Cà phê không có cây che bóng
- N2 Cà phê có cây che bóng hoặc trồng xen
- N3 Cao su
- N4 Điều
- N5 Hồ tiêu
- N6 Chè
- N7 Cây ăn quả
- N8 Cây lâu năm khác hoặc trồng hỗn hợp

**Nông nghiệp khác**
- N9 Cây hằng năm trên đất dốc hoặc nương rẫy đang canh tác (sắn, ngô, lúa nương, …)
- N10 Lúa nước, rau màu
- N11 Mía

**Khác**
- K1 Cỏ hoặc cây bụi
- K2 Đất trống, kể cả đất mới dọn chưa rõ dùng vào việc gì
- K3 Xây dựng hoặc đường
- K4 Mặt nước
- K5 Khai khoáng
- K6 Khác (ghi ở phần ghi chú)

Lưu ý: **cao su** luôn chọn N3, kể cả khi trồng lấy gỗ. Lý do là các định nghĩa khác nhau xếp cao su vào nhóm khác nhau, nên cần ghi riêng.

F1, F2 và F3 phải **thống nhất với nhau**. Ví dụ, F3 là N1–N11 thì F1 phải là "Không rừng".

## 7. Cây quyết định cho F1 và F2 (theo JRC Hình 7)

```
Trong khu vực có đất rừng không?
├─ Có → Ô giữa có nằm trong phần đất rừng không?
│        ├─ Có → Có dấu hiệu trồng (hàng lối, một loài, theo đường đồng mức)?
│        │        ├─ Có → RỪNG · Rừng trồng
│        │        └─ Không → RỪNG · Rừng nguyên sinh hoặc tái sinh tự nhiên
│        └─ Không (mảng không rừng trên 0,5 ha) → KHÔNG RỪNG · Đất không rừng nằm trong vùng rừng
└─ Không → Có cây không?
         ├─ Có → Đất dùng vào việc gì?
         │        ├─ Đô thị → KHÔNG RỪNG · Cây trong đô thị
         │        ├─ Nông nghiệp → Có dạng vườn trồng?
         │        │        ├─ Có → KHÔNG RỪNG · Cây phục vụ nông nghiệp
         │        │        └─ Không → KHÔNG RỪNG · Cây ngoài rừng
         │        └─ Khác → KHÔNG RỪNG · Cây ngoài rừng
         └─ Không → Có cây bụi không?
                  ├─ Có → KHÔNG RỪNG · Đất có cây gỗ khác
                  └─ Không → KHÔNG RỪNG · Không có cây hoặc cây bụi
```

Nếu không phân biệt được rừng tự nhiên với rừng trồng, chọn **rừng tự nhiên** (theo EUDR và JRC mục 2.3.4).

## 8. Quy trình

1. **Tập huấn** khoảng 1 giờ, dựa trên tài liệu này và các ví dụ.
2. **25 điểm hiệu chỉnh:** mỗi người gán độc lập, sau đó cả nhóm thảo luận các điểm khác nhau. Những điểm này không dùng trong kết quả.
3. **Giải đoán chính:** mỗi người gán các điểm được giao, độc lập, không giới hạn thời gian. Sau khoảng 15 điểm đầu, điều phối viên kiểm tra nhanh để phát hiện cách hiểu lệch (theo FRA RSS mục 3.3.1).
4. **Giải đoán lần hai:** một số điểm được giao cho người thứ hai mà người đó không biết nhãn đầu. Những điểm hai người khác nhau sẽ được thảo luận cho đến khi thống nhất. Nhãn ban đầu của từng người vẫn được giữ lại.
5. Khi chưa chắc, **chọn độ tin cậy thấp và ghi lý do**, thay vì bỏ qua. Không có cách trả lời "đúng" mà người điều phối mong đợi.
