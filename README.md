# FINAL PROJECT_MACHINE LEARNING

## Nghiên cứu bài 1
**SV1: Nguyễn Tấn Phát - 52000583**
## 1)	Tìm hiểu, so sánh các phương pháp Optimizer trong huấn luyện mô hình học máy:

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

***
## 2)	Tìm hiểu về Continual Learning và Test Production khi xây dựng một giải pháp học máy để giải quyết một bài toán nào đó.
### Continual Learning
**Khái niệm:**

Continual Learning (Học liên tục) là một quá trình trong đó một mô hình học từ các luồng dữ liệu mới mà không cần phải qua đào tạo lại. Khác với các phương pháp tiếp cận truyền thống, trong đó mô hình được đào tạo trên tập dữ liệu cố định, các mô hình học liên tục cập nhật lặp đi lặp lại các tham số của chúng để phản ánh các phân phối mới trong dữ liệu.

Quá trình mà mô hình không ngừng tự cải thiện bản thân bằng cách học từ thông tin mới nhất và điều chỉnh kiến thức khi có dữ liệu mới. Trong chu kỳ học máy liên tục, các mô hình có khả năng duy trì sự liên quan theo thời gian do khả năng chịu biến động của chúng.

**Quá trình của Continual Learning:**

1.	Initial training - Mô hình được đào tạo trên tập dữ liệu khởi đầu. Nó tạo ra một bộ tham số ban đầu dựa trên các mẫu mà nó gặp trong dữ liệu.
2.	Deployment - Mô hình được triển khai để thực hiện nhiệm vụ cụ thể. Trong giai đoạn này, dữ liệu mới liên quan đến nhiệm vụ và môi trường được thu thập.
3.	Data rehearsal - Mô hình được điều chỉnh thường xuyên, đều đặn bằng cách nhớ lại kinh nghiệm trước đó. Điều này giúp mô hình không quên thông tin đã học trước đó trong quá trình huấn luyện trên dữ liệu mới.
4.	Continuous learning strategy - Áp dụng một chiến lược học liên tục để thích ứng và cải thiện hiệu suất của mô hình. Chiến lược này giúp mô hình duy trì tính liên quan theo thời gian và thích ứng với sự biến động trong dữ liệu và môi trường.
5.	Revaluation and monitoring - Hiệu suất của mô hình được đánh giá định kỳ, bao gồm độ chính xác, khả thi, hành vi thực tế và độ chệch. Quá trình này giúp theo dõi sự tiến triển và xác định cần điều chỉnh gì để cải thiện mô hình.

**Thách thức:**

Continual Learning đối mặt với một loạt các thách thức, bao gồm việc đối diện với hiện tượng *Catastrophic Forgetting*. Trong quá trình đào tạo với dữ liệu mới, mô hình có thể không chỉ quên thông tin của các nhiệm vụ trước đó mà còn làm giảm độ chính xác đối với chúng. 
Ngoài ra, một thách thức khác là *Preserving Knowledge*, yêu cầu mô hình phải có khả năng bảo toàn kiến thức đã học trước đó khi tiếp tục học từ dữ liệu mới. 
Cuối cùng, việc *tích hợp dữ liệu mới (Incorporating new data)* cũng là một vấn đề quan trọng. Mô hình cần có khả năng tích hợp dữ liệu mới một cách linh hoạt mà không gây ảnh hưởng đáng kể đến khả năng dự đoán trên các nhiệm vụ đã được thực hiện trước đó.

**Tại sao sử dụng Continual Learning?**

Nguyên nhân chính để thực hiện việc điều chỉnh liên tục cho mô hình máy học là để đảm bảo rằng mô hình có thể thích ứng nhanh chóng với sự biến động trong phân phối dữ liệu. 

Các trường hợp sử dụng điển hình bao gồm những tình huống mà sự thay đổi có thể xảy ra đột ngột và mà việc thích ứng linh hoạt là quan trọng. Ví dụ cho trường hợp cần sự thích ứng nhanh chóng với sự kiện thương mại lớn có thể là các chương trình khuyến mãi đặc biệt như ưu đãi giảm giá mùa lễ, sự kiện quảng bá sản phẩm độc đáo hoặc các sự kiện khuyến mãi đặc biệt không theo chu kỳ cố định. Những tình huống này thường xuyên xuất hiện không cố định trong lịch trình và đòi hỏi mô hình phải thích ứng ngay lập tức để đưa ra dự đoán chính xác và phản hồi nhanh chóng cho người dùng.

**Stateful Training và Stateless Retraining**

<img src="https://i.imgur.com/w4PRIoQ.png">

  •	Huấn luyện có trạng thái (Stateful Training): mô hình giữ lại kiến thức từ các nhiệm vụ trước và tiếp tục học mà không quên chúng. Điều này đòi hỏi các cơ chế để tránh quên đột ngột.
  
  •	Huấn luyện lại không trạng thái (Stateless Retraining): mô hình được huấn luyện trên các nhiệm vụ mới mà không giữ lại kiến thức của các nhiệm vụ trước. Phương pháp này có nguy cơ quên thông tin của các nhiệm vụ cũ.

Chênh lệch giữa hai khái niệm này có thể được nhận biết bằng cách mô tả quá trình đào tạo Stateful và Stateless. Trong Stateful Training, mô hình duy trì trạng thái của nó qua các nhiệm vụ khác nhau, trong đó trạng thái có thể chứa các tham số, trọng số mạng nơ-ron hoặc các giá trị khác liên quan đến trạng thái hiện tại của mô hình. Khi chuyển đổi giữa các nhiệm vụ, mô hình sử dụng trạng thái hiện tại để hỗ trợ quá trình học và giữ lại kiến thức đã học từ trước.

Ngược lại, *Stateless Retraining* không bảo toàn trạng thái giữa các nhiệm vụ. Thay vào đó, khi chuyển từ một nhiệm vụ sang nhiệm vụ mới, mô hình bắt đầu lại quá trình đào tạo từ trạng thái ban đầu hoặc một trạng thái ngẫu nhiên. Phương pháp này có thể dẫn đến hiện tượng quên (catastrophic forgetting), khi mô hình mất kiến thức đã học khi đối mặt với dữ liệu mới.

Sự chọn lựa giữa *Stateful Training* và *Stateless Retraining* phụ thuộc vào yêu cầu cụ thể của ứng dụng và ngữ cảnh đào tạo. Có nhiều kỹ thuật được áp dụng để giải quyết thách thức của cả hai phương pháp, bao gồm memory replay (lưu trữ và tái sử dụng dữ liệu từ quá khứ) và các kỹ thuật regularization để giảm thiểu nguy cơ quên kiến thức đã học.

**Đo lường sự thay đổi của dữ liệu**

Để đánh giá giá trị của dữ liệu mới, một phương pháp là huấn luyện một mô hình có cùng cấu trúc trên dữ liệu từ ba khoảng thời gian khác nhau và sau đó kiểm thử mỗi mô hình trên dữ liệu hiện tại được gán nhãn. Nếu quan sát được rằng việc để mô hình lỗi thời trong vòng 3 tháng dẫn đến sự chênh lệch 10% trong độ chính xác của dữ liệu kiểm thử hiện tại, thì việc huấn luyện lại nên được thực hiện trong khoảng thời gian ít hơn 3 tháng.

**Ưu và nhược điểm**

•	Ưu Điểm:
  1.	Khả năng khái quát hóa và dự đoán tốt hơn: mô hình có khả năng khái quát hóa thông tin và dự đoán tốt hơn nhờ việc tích lũy kiến thức theo thời gian. Điều này giúp mô hình đưa ra dự đoán chính xác dựa trên trải nghiệm và thông tin lịch sử.
  2.	Giữ lại và xây dựng dựa trên kiến thức đã học: Continual learning giúp mô hình giữ lại và xây dựng kiến thức từ các kinh nghiệm trước đó, tạo nền tảng vững chắc cho quá trình học và dự đoán trong tương lai.
  3.	Thích ứng tốt với dữ liệu và kiến thức mới: có khả năng thích ứng linh hoạt với dữ liệu và kiến thức mới, giúp mô hình duy trì độ chính xác và hiệu suất khi có sự thay đổi trong môi trường hoặc dữ liệu.

•	Nhược điểm:
  1.	Khó quản lý các phiên bản mô hình khác nhau: quản lý nhiều phiên bản mô hình có thể là một thách thức, đặc biệt khi mô hình phải điều chỉnh liên tục để thích ứng với dữ liệu mới và kiến thức.
  2.	Cần xử lý liên tục dữ liệu mới, dễ bị ảnh hưởng bởi dữ liệu trôi dạt: việc xử lý liên tục dữ liệu mới có thể đòi hỏi nhiều công sức và tài nguyên, đồng thời mô hình có thể bị ảnh hưởng bởi sự biến động và nhiễu từ dữ liệu mới.
- - -
### Test Production

**Khái niệm:**

Test Production là giai đoạn quyết định, mô hình học máy sau khi được huấn luyện, trải qua quá trình đánh giá chặt chẽ trong một bối cảnh thực tế. Mục tiêu chính là xác định hiệu suất của mô hình và khả năng đáp ứng yêu cầu kinh doanh khi triển khai trong môi trường sản xuất.

**Các bước chính trong Test Production:**
  1.	Chuẩn bị bộ dữ liệu kiểm thử - Bộ dữ liệu này cần chia sẻ các đặc tính phân phối tương tự với dữ liệu huấn luyện và hoàn toàn độc lập để huấn luyện mô hình.
  2.	Triển khai mô hình - Triển khai mô hình đã huấn luyện vào môi trường thực tế giống hệt với môi trường production.
  3.	Kiểm thử - Cho mô hình dự đoán trên tập dữ liệu kiểm tra và thu thập các metric (accuracy, recall, precision, F1 score,...).
  4.	Phân tích và đánh giá - Các chỉ số thu thập được được phân tích so với ngưỡng mong đợi và yêu cầu thực tế để đánh giá hiệu quả của mô hình.
  5.	Tinh chỉnh và cải tiến - Dựa trên phân tích, các điều chỉnh cần thiết được thực hiện đối với quy trình xây dựng và huấn luyện mô hình.

**Đánh giá trước khi triển khai**

Có hai phương pháp phổ biến được sử dụng là Test Splits và Backtesting để đánh giá hiệu suất mô hình:

***Test Splits:*** Sử dụng các tập kiểm thử tĩnh để so sánh với một điểm cơ sở và thực hiện các kiểm thử lại. Tập kiểm thử thường là tĩnh để cung cấp một điểm chuẩn đáng tin cậy để so sánh giữa các mô hình. *Tuy nhiên*, hiệu suất tốt trên một tập kiểm thử cụ thể không đảm bảo hiệu suất tốt trong môi trường sản xuất do sự thay đổi trong phân phối dữ liệu hiện tại.

***Backtesting:*** Sử dụng dữ liệu mới nhất, chưa được mô hình thấy trong quá trình huấn luyện, để kiểm thử hiệu suất. *Tuy nhiên*, cần chú ý đến các yếu tố như độ trễ, hành vi người dùng đối với mô hình và tính đúng đắn của tích hợp hệ thống để đảm bảo an toàn khi triển khai rộng rãi. Mặc dù backtesting cung cấp cái nhìn về hiệu suất trên dữ liệu mới, nhưng quan sát kỹ thuật này là quan trọng để đảm bảo mô hình hoạt động hiệu quả trong điều kiện thực tế.

**Testing trong Production Strategies:**
  1.	Shadow Deployment
  - Mô tả: Một phiên bản đối phó hoặc "shadow" của mô hình được triển khai cùng với mô hình hiện tại trên sản xuất. Nó xử lý dữ liệu sản xuất thực tế, nhưng các dự đoán của nó không được sử dụng để đưa ra quyết định.
  - Ưu điểm: đây là cách an toàn nhất để triển khai mô hình của bạn. Ngay cả khi mô hình mới của bạn có lỗi, dự đoán sẽ không được đưa ra.
  - Hạn chế: 
    +	Không thể sử dụng kỹ thuật này khi đo lường hiệu suất của mô hình phụ thuộc vào việc quan sát cách người dùng tương tác với các dự đoán.
    +	Kỹ thuật này tốn kém khi chạy vì nó tăng gấp đôi số lượng dự đoán và do đó số lượng tính toán cần thiết.

  2.	A/B Testing
  - Mô tả: Triển khai mô hình đối thủ (mô hình B) đồng thời với mô hình hiện tại (mô hình A) và định tuyến một phần trăm lưu lượng đến mô hình đối thủ là một chiến lược để đánh giá hiệu suất giữa hai mô hình. Dự đoán từ mô hình đối thủ được hiển thị cho người dùng, và sau đó, sử dụng theo dõi và phân tích kết quả dự đoán từ cả hai mô hình để xác định xem hiệu suất của mô hình đối thủ có sự cải thiện thống kê so với mô hình hiện tại không.
    Trong một số trường hợp sử dụng không tương thích với việc chia lưu lượng và triển khai nhiều mô hình cùng một lúc, chiến lược thử nghiệm A/B có thể được thực hiện theo thời gian. Điều này bao gồm việc chia lưu lượng theo chu kỳ thời gian, chẳng hạn như một ngày cho mô hình A và ngày tiếp theo cho mô hình B. Quan trọng là phân chia lưu lượng phải là một thử nghiệm ngẫu nhiên thực sự, đảm bảo rằng việc chọn mô hình A hoặc B không bị thiên lệch.
    Nếu có bất kỳ thiên lệch lựa chọn nào trong quá trình phân chia lưu lượng (ví dụ: người dùng máy tính nhận mô hình A và người dùng di động nhận mô hình B), kết quả của thử nghiệm sẽ không chính xác. Để đảm bảo tính chính xác của kết quả, thử nghiệm phải chạy đủ lâu để thu thập đủ mẫu và đạt được độ tin cậy thống kê đáng tin cậy.
  - Ưu điểm: 
    +	Dễ hiểu và cho phép so sánh trực tiếp hiệu suất mô hình dưới điều kiện thực tế. Giúp đưa ra quyết định dựa trên dữ liệu.
    +	Dự đoán được cung cấp cho người dùng nên kỹ thuật này cho phép bạn nắm bắt đầy đủ cách người dùng phản ứng với các mô hình khác nhau.
  - Hạn chế: 
    +	Kém an toàn hơn so với Shadow Deployment
    +	Bạn phải đối mặt với giả định nhiều rủi ro.
    +	Trong trường hợp cần phản ứng nhanh với biến động trong dữ liệu, việc chờ đợi đủ dữ liệu có thể là một hạn chế. Bạn có thể không thể đưa ra quyết định ngay lập tức nếu cần sự linh hoạt và ứng phó nhanh chóng.
  3. Canary Release
  - Mô tả: Mô hình mới được triển khai từ từ cho một tập con nhỏ người dùng hoặc lưu lượng. Hiệu suất được giám sát chặt chẽ, và nếu thành công, quá trình triển khai mở rộng ra một đối tượng lớn hơn.
  - Ưu điểm: 
    +	Cho phép tiếp cận dần dần để xác định vấn đề sớm. Giảm thiểu ảnh hưởng của vấn đề có thể xảy ra trên quy mô lớn.
    +	Nếu kết hợp với A/B Testing, nó cho phép bạn thay đổi linh hoạt lượng lưu lượng truy cập mà mỗi mô hình đang sử dụng.
  -	Hạn chế: Nếu việc triển khai không được giám sát cẩn thận, có thể xảy ra những sự cố không mong muốn. Đây có thể coi là lựa chọn an toàn nhất, tuy nhiên, nó cũng dễ dàng quay trở lại trạng thái trước đó nếu cần thiết.
  4.	Interleaving Experiments
  -	Mô tả: Trong thử nghiệm A/B (A/B Testing), một người dùng sẽ nhận được dự đoán từ mô hình A hoặc mô hình B. Khi xen kẽ, một người dùng sẽ nhận được dự đoán xen kẽ từ cả mô hình A và mô hình B. Sau đó, chúng tôi theo dõi hiệu quả hoạt động của từng mô hình bằng cách đo lường mức độ ưu tiên của người dùng với từng mô hình dự đoán của mô hình (ví dụ: người dùng nhấp nhiều hơn vào đề xuất từ mô hình B)
  -	Ưu điểm: Cung cấp so sánh trực tiếp giữa kết quả mô hình trong các kịch bản thực tế. Giảm thiểu độ chệch từ các yếu tố thời gian hoặc người dùng cụ thể.
  -	Hạn chế: 
    +	Việc triển khai phức tạp hơn thử nghiệm A/B - nếu một trong các mô hình xen kẽ mất quá nhiều thời gian để phản hồi hoặc không thành công
    +	Nó tăng gấp đôi sức mạnh tính toán cần thiết vì mọi yêu cầu đều nhận được dự đoán từ nhiều mô hình.
  5.	Bandits
  - Mô tả: Bandits là một thuật toán theo dõi hiệu suất hiện tại của từng biến thể mô hình và đưa ra quyết định linh hoạt đối với mọi yêu cầu về việc nên sử dụng mô hình có hiệu suất cao nhất cho đến nay (tức là khai thác kiến thức hiện tại) hay thử bất kỳ mô hình nào khác để có thêm thông tin về chúng (tức là khám phá xem một trong các mô hình khác thực sự tốt hơn).
  - Ưu điểm:
    +	Bandits ít mất dữ liệu hơn A/B Testing để xác định mô hình nào tốt hơn. Ngoài ra, tốc độ hội tụ sẽ nhanh hơn và an toàn hơn.
    +	Sử dụng dữ liệu hiệu quả hơn đồng thời giảm thiểu chi phí cơ hội của bạn (opportunity cost)
  -	Hạn chế:
    +	Khó thực hiện hơn do phải truyền phải hồi vào thuật toán một cách liên tục.
    +	Không an toàn như Shadow Deployment vì phải đối mặt với thách thức chiếm lưu lượng truy cập trực tiếp.
