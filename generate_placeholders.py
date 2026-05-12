# generate_placeholders.py

from PIL import Image, ImageDraw, ImageFont
import qrcode
import matplotlib.pyplot as plt
import numpy as np

# -------------------------
# 1️⃣ Hospital Logo Placeholder
# -------------------------
logo = Image.new("RGB", (150, 150), color=(30, 144, 255))  # blue square
draw = ImageDraw.Draw(logo)
draw.text((20, 60), "HOSPITAL", fill="white")
logo.save("hospital_logo.png")

# -------------------------
# 2️⃣ SHAP Plot Placeholder
# -------------------------
plt.figure(figsize=(6,4))
features = ["radius_mean","texture_mean","perimeter_mean","area_mean"]
importance = [0.4,0.3,0.2,0.1]
plt.barh(features, importance, color='skyblue')
plt.xlabel("SHAP Importance")
plt.title("SHAP Feature Importance")
plt.tight_layout()
plt.savefig("shap_plot.png")
plt.close()

# -------------------------
# 3️⃣ Doctor Signature Placeholder
# -------------------------
sig = Image.new("RGBA", (300, 100), color=(255,255,255,0))  # transparent
draw = ImageDraw.Draw(sig)
draw.text((10, 40), "Dr. John Doe", fill="black")
sig.save("doctor_signature.png")

# -------------------------
# 4️⃣ Verification QR Code
# -------------------------
data = "https://example.com/verify"
qr = qrcode.QRCode(
    version=1,
    box_size=6,
    border=4
)
qr.add_data(data)
qr.make(fit=True)
img_qr = qr.make_image(fill_color="black", back_color="white")
img_qr.save("verification_qr.png")

print("All placeholder images generated successfully!")
