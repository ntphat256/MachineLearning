# FINAL PROJECT_MACHINE LEARNING

## Nghiên cứu bài 1
**SV1: Nguyễn Tấn Phát - 52000583**
#### 1)	Tìm hiểu, so sánh các phương pháp Optimizer trong huấn luyện mô hình học máy:

### Optimizer
Optimizer (thuật toán tối ưu) là các thuật toán dùng để điều chỉnh các tham số của mô hình (weights, bias) trong quá trình huấn luyện để giảm thiểu hàm mất mát (loss function), cải thiện hiệu suất của mô hình.
### Các thuật toán tối
#### Gradient Descent (GD)
Trong các bài toán tối ưu, chúng ta thường cố gắng tìm giá trị nhỏ nhất của một hàm số cụ thể. Để đạt được giá trị nhỏ nhất, chúng ta thường kiểm tra nếu đạo hàm của hàm số bằng 0, vì điều này thường là dấu hiệu của một điểm cực tiểu. Tuy nhiên, không phải lúc nào việc này cũng dễ dàng, đặc biệt là khi làm việc với các hàm số có nhiều biến. Đạo hàm của các hàm số đa biến thường phức tạp và đôi khi không thể tính toán được. Do đó, thay vì điều này, chúng ta thường tìm điểm gần với điểm cực tiểu nhất và xem đó như là nghiệm của bài toán tối ưu.
- **GD cho hàm một biến:**
  - Hướng di chuyển: Nếu đạo hàm f'(x_t) > 0, điểm x_t nằm về phía bên phải so với điểm cực tiểu x*, và ngược lại. Để đưa x_(t+1) gần hơn x*, chúng ta cần di chuyển x_t về phía bên trái, tức là về phía âm. Điều này được thực hiện bằng cách di chuyển theo chiều ngược dấu với đạo hàm: x_(t+1) = x_t + ∆
  - Công thức cập nhật: x_(t+1) = x_t – η f'(x_t) , trong đó η (learning rate) là một số dương quyết định tốc độ học. Dấu trừ ở đây là để di chuyển ngược với hướng đạo hàm để tiến tới điểm cực tiểu.
  - Learning rate η quan trọng để kiểm soát tốc độ học của thuật toán. Nếu quá lớn, có thể dẫn đến overshooting hoặc khó hội tụ. Nếu quá nhỏ, có thể làm chậm quá trình học.
- **GD cho hàm nhiều biến:**
  - Bước 1: Chọn một vector giá trị ban đầu x_0 để bắt đầu quá trình tối ưu hóa
  - Bước 2: Tính đạo hàm của hàm số tại điểm x_t, ký hiệu là ∇f(x_t) (hình tam giác ngược đọc là nabla).
  - Bước 3: Di chuyển điểm x_t theo chiều ngược với đạo hàm x_(t+1) = x_t – η ∇ f(x_t)

Trong đó, η là learning rate, quyết định kích thước bước di chuyển.

*Quy tắc cần nhớ*: luôn luôn đi ngược hướng với đạo hàm

=> Quá trình này tiếp tục cho đến khi đạt được một điểm gần đúng của nghiệm tối ưu hoặc khi đạt đến một số lần lặp tối đa được đặt trước. Tính chất chính của Gradient Descent là sự lặp lại liên tục và di chuyển theo chiều giảm của hàm số để tiến tới điểm cực tiểu.

#### Stochastic Gradient Descent (SGD)
Stochastic Gradient Descent (SGD) là một biến thể của thuật toán Gradient Descent, được sử dụng rộng rãi trong học máy. Nó hoạt động bằng cách cập nhật các tham số mô hình dựa trên gradient của hàm mất mát (loss function), tính toán trên một mini-batch ngẫu nhiên từ dữ liệu huấn luyện thay vì toàn bộ tập dữ liệu.

**Mô tả thuật toán SGD:**
- Initialization: Khởi tạo ngẫu nhiên các tham số của mô hình.
- Set Parameters: Xác định số lần lặp và tốc độ học (alpha) để cập nhật tham số.
- Lặp lại các bước sau cho đến khi mô hình hội tụ hoặc đạt số lần lặp tối đa:
    +	Xáo trộn tập dữ liệu huấn luyện để tạo ra tính ngẫu nhiên.
    +	Lặp lại từng ví dụ huấn luyện theo thứ tự được xáo trộn.
    +	Tính toán độ dốc của hàm chi phí đối với các tham số mô hình bằng cách sử dụng mẫu (hoặc batch) đào tạo hiện tại.
    +	Cập nhật các tham số mô hình bằng cách thực hiện một bước theo hướng gradient âm, được chia tỷ lệ theo tốc độ học.
    +	Đánh giá các tiêu chí hội tụ, chẳng hạn như sự khác biệt trong hàm chi phí giữa các lần lặp của gradient.
- Trả về các tham số được tối ưu hóa: Sau khi đáp ứng các tiêu chí hội tụ hoặc đạt đến số lần lặp tối đa, hãy trả về các tham số mô hình được tối ưu hóa.

#### Momentum
Gradient Descent với Momentum, để di chuyển đến vị trí mới của nghiệm (tức là hòn bi), chúng ta cần tính toán lượng thay đổi tại thời điểm t. Nếu ta giả sử đại lượng này tương đương với vận tốc v_t trong bối cảnh vật lý, thì vị trí mới của hòn bi sẽ được cập nhật theo công thức và dấu trừ ở đây thể hiện việc di chuyển ngược với độ dốc: 

θ_(t+1) = θ_t – v_t

Để tính v_t, ta cần kết hợp thông tin về độ dốc (∇θJ(θ)) và vận tốc trước đó (v_(t-1), mà ta giả sử là vận tốc ban đầu v_0 = 0). Cách đơn giản nhất là tổng hợp hai đại lượng này với trọng số, được biểu diễn bằng phương trình: 

v_t = γ v_(t-1) + η ∇ θ f(θ)

Trong đó: 
- γ : thường được chọn trong khoảng 0.9
- v_t: vận tốc tại thời điểm trước đó
- ∇θf(θ): độ dốc tại điểm trước đó. Vị trí mới của nghiệm sau đó được cập nhật bằng cách trừ đi v_t: θ = θ – v_t

#### Adagrad
Adagrad (Adaptive Gradient) là một thuật toán tối ưu hóa được sử dụng để tối ưu hóa quá trình huấn luyện của các mạng nơ-ron. Khả năng thích ứng với từng tham số cụ thể bằng cách điều chỉnh tốc độ học tập tại mỗi lần lặp dựa trên biến động của gradient của tham số đó.

Ở bước đầu tiên, Adagrad tính gradient của mỗi tham số giống như các thuật toán giảm độ dốc khác. Tuy nhiên, điểm đặc biệt của Adagrad là nó sử dụng lịch sử của gradient để cập nhật tốc độ học tập. Nó giữ một bộ nhớ tích lũy các bình phương gradient đã tính toán trước đó.
Khi một tham số có gradient thay đổi nhiều, bình phương gradient tương ứng trong lịch sử sẽ càng lớn. Điều này dẫn đến việc giảm tốc độ học tập cho tham số đó, giúp ổn định quá trình học trên các đặc trưng có biến động cao. Ngược lại, đối với các tham số có gradient biến động ít, tốc độ học tập sẽ được giữ ở mức cao hơn, giúp mô hình học tốt hơn trên các đặc trưng có biến động thấp.

Mặc dù Adagrad có nhược điểm về việc giảm tốc độ học tập quá nhanh khi tích lũy gradient lớn, nhưng nó vẫn là một công cụ hữu ích trong quá trình huấn luyện mạng nơ-ron, đặc biệt là khi đối mặt với dữ liệu có đặc trưng thưa thớt hoặc dày đặc.

Công thức cập nhật trọng số trong thuật toán Adagrad sử dụng các yếu tố như alpha(t) để đại diện cho tốc độ học tập thay đổi ở mỗi lần lặp, n là hằng số, và E là giá trị nhỏ để tránh việc chia cho 0:

<img src="https://i.imgur.com/YHjvTl8.jpg">

#### RMSProp
RMSProp (Root Mean Squared Propagation) là một phần mở rộng cho thuật toán tối ưu Gradient Descent. RMSprop giải quyết vấn đề về tỷ lệ học giảm dần của Adagrad. RMSprop là một biến thể của Adagrad được thiết kế để cải thiện hiệu suất trên dữ liệu và mô hình có biến động gradient lớn.

Trong quá trình huấn luyện, RMSprop thực hiện việc giảm tốc độ học tập cho mỗi tham số theo tỷ lệ của bình phương trung bình di động của gradient. Cụ thể, nó sử dụng một trọng số giảm dần, thường được ký hiệu là beta, để đánh giá mức độ giảm của gradient trước đó trong trung bình di động.

Công thức cập nhật trọng số:

<img src="https://i.imgur.com/O4SHB1p.jpg">

Trong đó:
- θ_t : tham số tại bước lặp thứ t
- g_t : gradient tại bước lặp thứ t
- E[[g]^2]_t : RMSProp của gradient tính đến bước lặp thứ t
- α : tốc độ học
- ϵ : một giá trị nhỏ để tránh chia cho 0.

Bằng cách này, RMSprop giúp ổn định và thích ứng tốc độ học tập theo biến động của gradient, đồng thời tránh vấn đề của Adagrad khi giảm quá nhanh tốc độ học tập.

#### Adam
Adam (Adaptive Moment Estimation) là một thuật toán tối ưu hóa phổ biến trong huấn luyện mạng nơ-ron, được thiết kế để cải thiện hiệu suất và tốc độ hội tụ. Adam kết hợp hai ý tưởng chính từ động lượng (momentum) và RMSProp để tối ưu hóa quá trình học.

Trong quá trình huấn luyện, Adam duy trì một trung bình trượt của gradient, biểu thị giá trị trung bình và phương sai của chúng. Phần trung bình trượt gradient trong Adam đóng vai trò như một "bộ nhớ" động lượng, giúp theo dõi hướng di chuyển ngay cả khi gradient thay đổi nhỏ. Điều này giúp tăng tốc quá trình học và giảm thiểu dao động không mong muốn.

Phần phương sai trong Adam thích ứng tốc độ học riêng biệt cho từng tham số. Nhờ đó, Adam tự động điều chỉnh tốc độ học tập tại mỗi tham số, tạo điều kiện cho việc huấn luyện hiệu quả trên các đặc trưng có biến động lớn khác nhau.

Đây là một phương pháp tối ưu hóa linh hoạt và mạnh mẽ, giúp mô hình nhanh chóng hội tụ đến giải pháp tối ưu trong quá trình huấn luyện.

#### So sánh các Optimizer
 
|       Optimizer       |      Ưu điểm            | Nhược điểm                            |
| --------------------|-------------------|-------------------------------------------------|
|    GD          |- Cơ bản, dễ hiểu, giải quyết được các vấn đề tối ưu model neural network bằng cập nhật trọng số sau mỗi vòng lặp.|- Phụ thuộc vào nghiệm khởi tạo ban đầu và learning rate.<br/>- Tốc độ học quá lớn sẽ khiến thuật toán không hội tụ, ảnh hưởng đến tốc độ training.|
|     SGD         |- Tính toán hiệu quả, dễ thực hiện.<br/> - Hiệu quả đối với các tập dữ liệu lớn với không gian đặc trưng nhiều chiều.|- Yêu cầu nhiều lần lặp hơn (thường sẽ yêu cầu lặp nhiều hơn Gradient Descent để hội tụ).<br/> - Nhạy cảm với tốc độ học ban đầu.|
|     Momentum         | - Giải quyết được vấn đề Gradient Descent không tiến được tới điểm global minimum mà chỉ dừng ở local minimum.|-	Mất nhiều thời gian giao động qua lại trước khi dừng hẳn, khi tới gần đích.  |
|     Adagrad         |- Hiệu quả với tập dữ liệu thưa thớt.<br/> - Tỷ lệ học thích ứng cho mỗi tham số.|- Cần lưu trữ lịch sử gradient của tất cả tham số, nên có thể tốn kém về bộ nhớ và tính toán.<br/> - Giảm hiệu quả sau khi hội tụ (có thể tiếp tục giảm tốc độ học, dẫn đến training bị chậm)|
|     RMSProp         |- Tốc độ học thích ứng trên mỗi tham số giúp hạn chế sự tích lũy độ dốc.<br/> - Hiệu quả đối với các mục tiêu không cố định.|- Có thể có tốc độ hội tụ chậm trong một số trường hợp (ví dụ: tốc độ học quá cao hoặc quá thấp sẽ gây ra vấn đề về tốc độ hội tụ,...)  |
|       Adam       |- Áp dụng được cho các tập dữ liệu lớn và mô hình nhiều chiều.<br/> - Khái quát hóa tốt.|- Phải điều chỉnh cẩn thận các hyperparameter.  |
