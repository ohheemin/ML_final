import cv2
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import rclpy
from rclpy.node import Node
from std_msgs.msg import Int32
from rclpy.qos import QoSProfile

# Conv-BN-SiLU
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU()
        )
    
    def forward(self, x):
        return self.block(x)

# C2f Block
class C2fBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_blocks=1):
        super(C2fBlock, self).__init__()
        hidden_channels = out_channels // 2
        self.stem = ConvBlock(in_channels, hidden_channels, kernel_size=1, padding=0)

        self.blocks = nn.Sequential(*[
            ConvBlock(hidden_channels, hidden_channels) for _ in range(num_blocks)
        ])

        self.concat = ConvBlock(in_channels + hidden_channels, out_channels, kernel_size=1, padding=0)

    def forward(self, x):
        y1 = self.stem(x)
        y2 = self.blocks(y1)
        out = torch.cat([x, y2], dim=1)
        return self.concat(out)

class YOLOv8Classifier(nn.Module):
    def __init__(self, num_classes):
        super(YOLOv8Classifier, self).__init__()
        self.backbone = nn.Sequential(
            ConvBlock(3, 32, kernel_size=3),          # 64x64
            C2fBlock(32, 64, num_blocks=1),           
            nn.MaxPool2d(2),                          # 32x32

            C2fBlock(64, 128, num_blocks=2),          
            nn.MaxPool2d(2),                          # 16x16

            C2fBlock(128, 256, num_blocks=2),         
            nn.MaxPool2d(2),                          # 8x8

            ConvBlock(256, 512, kernel_size=1, padding=0),
        )
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.backbone(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

class RealTimeClassifierNode(Node):
    def __init__(self):
        super().__init__('real_time_classifier')
        
        self.class_to_idx = {'50c0d': 0, '50c15d': 1, '50c30d': 2, '50c45d': 3, '50c60d': 4, '50c75d': 5, '50c90d': 6}
        self.idx_to_angle = {0: 0, 1: 15, 2: 30, 3: 45, 4: 60, 5: 75, 6: 90}

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = YOLOv8Classifier(num_classes=len(self.class_to_idx)).to(self.device)
        self.model.load_state_dict(torch.load("/home/ohheemin/mcln/0616_augmented.pth", map_location=self.device))
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3),
        ])
        qos = QoSProfile(depth=10)
        self.publisher_ = self.create_publisher(Int32, '/prediction_angle', qos)

        self.cap = cv2.VideoCapture(2)
        if not self.cap.isOpened():
            self.get_logger().error("카메라 열기 실패")
            exit()

        self.timer = self.create_timer(0.1, self.timer_callback)

    def timer_callback(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img)
        input_tensor = self.transform(pil_img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(input_tensor)
            _, pred_idx = torch.max(outputs, 1)
            angle = self.idx_to_angle[pred_idx.item()]

        msg = Int32()
        msg.data = angle
        self.publisher_.publish(msg)

        cv2.putText(frame, f'Prediction: {angle} deg', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Real-time Classification', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            self.cap.release()
            cv2.destroyAllWindows()
            rclpy.shutdown()

def main(args=None):
    rclpy.init(args=args)
    node = RealTimeClassifierNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
