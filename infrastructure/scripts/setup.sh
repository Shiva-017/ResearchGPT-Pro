#!/bin/bash
# ============================================================
# ResearchGPT Pro — One-time EC2 setup
# ============================================================
# Run this ONCE on a fresh Ubuntu EC2 instance:
#   chmod +x infrastructure/scripts/setup.sh
#   ./infrastructure/scripts/setup.sh
#
# Prerequisites:
#   - Ubuntu 22.04+ EC2 instance (t2.micro or larger)
#   - SSH access
#   - .env file ready with API keys
# ============================================================

set -e

APP_DIR=~/ResearchGPT-Pro
echo "============================================"
echo "ResearchGPT Pro — EC2 Setup"
echo "============================================"

# ── 1. System packages ──────────────────────────────────────
echo ""
echo "=== Installing system packages ==="
sudo apt-get update -qq
sudo apt-get install -y -qq python3 python3-pip python3-venv nginx git curl

# ── 2. Node.js 20 ───────────────────────────────────────────
echo ""
echo "=== Installing Node.js 20 ==="
if ! command -v node &> /dev/null || [[ $(node -v | cut -d. -f1 | tr -d v) -lt 18 ]]; then
    curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
    sudo apt-get install -y -qq nodejs
fi
echo "Node.js: $(node -v)"
echo "npm: $(npm -v)"

# ── 3. Clone repo (if not already there) ────────────────────
echo ""
echo "=== Setting up repository ==="
if [ ! -d "$APP_DIR" ]; then
    echo "Cloning repository..."
    git clone https://github.com/yourusername/ResearchGPT-Pro.git $APP_DIR
else
    echo "Repository already exists, pulling latest..."
    cd $APP_DIR
    git pull origin main
fi
cd $APP_DIR

# ── 4. Create data directories ──────────────────────────────
echo ""
echo "=== Creating data directories ==="
mkdir -p backend/data/{raw,pdfs,processed,checkpoints,logs}

# ── 5. Python dependencies ──────────────────────────────────
echo ""
echo "=== Installing Python dependencies ==="
pip3 install -r requirements.txt --quiet

# ── 6. Frontend dependencies + build ────────────────────────
echo ""
echo "=== Installing and building frontend ==="
cd frontend
npm ci --production=false
npm run build
cd ..

# ── 7. .env check ───────────────────────────────────────────
echo ""
if [ ! -f .env ]; then
    echo "⚠️  WARNING: .env file not found!"
    echo "Create it with: nano $APP_DIR/.env"
    echo ""
    echo "Required keys:"
    echo "  OPENAI_API_KEY=sk-..."
    echo "  PINECONE_API_KEY=..."
    echo "  COHERE_API_KEY=..."
    echo "  PINECONE_INDEX_NAME=research-papers"
    echo ""
else
    echo "✅ .env file found"
fi

# ── 8. Systemd services ─────────────────────────────────────
echo ""
echo "=== Installing systemd services ==="
sudo cp infrastructure/systemd/researchgpt-backend.service /etc/systemd/system/
sudo cp infrastructure/systemd/researchgpt-frontend.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable researchgpt-backend
sudo systemctl enable researchgpt-frontend

# ── 9. Nginx ────────────────────────────────────────────────
echo ""
echo "=== Configuring Nginx ==="
sudo cp infrastructure/nginx/researchgpt.conf /etc/nginx/sites-available/researchgpt
sudo ln -sf /etc/nginx/sites-available/researchgpt /etc/nginx/sites-enabled/researchgpt
sudo rm -f /etc/nginx/sites-enabled/default
sudo nginx -t
sudo systemctl restart nginx
sudo systemctl enable nginx

# ── 10. Configure swap (critical for t2.micro) ──────────────
echo ""
echo "=== Configuring 1GB swap ==="
if [ ! -f /swapfile ]; then
    sudo fallocate -l 1G /swapfile
    sudo chmod 600 /swapfile
    sudo mkswap /swapfile
    sudo swapon /swapfile
    echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
    echo "✅ 1GB swap enabled"
else
    echo "Swap already configured"
fi

# ── 11. Start services ──────────────────────────────────────
echo ""
echo "=== Starting services ==="
sudo systemctl start researchgpt-backend
sudo systemctl start researchgpt-frontend

echo ""
echo "=== Waiting for services to start ==="
sleep 5

# ── 12. Verify ──────────────────────────────────────────────
echo ""
echo "============================================"
echo "SETUP COMPLETE"
echo "============================================"
echo ""
echo "Services:"
sudo systemctl is-active researchgpt-backend && echo "  ✅ Backend running" || echo "  ❌ Backend failed"
sudo systemctl is-active researchgpt-frontend && echo "  ✅ Frontend running" || echo "  ❌ Frontend failed"
sudo systemctl is-active nginx && echo "  ✅ Nginx running" || echo "  ❌ Nginx failed"
echo ""
echo "Access: http://$(curl -s http://169.254.169.254/latest/meta-data/public-ipv4 2>/dev/null || echo 'YOUR_EC2_IP')"
echo ""
echo "Useful commands:"
echo "  sudo systemctl status researchgpt-backend"
echo "  sudo systemctl status researchgpt-frontend"
echo "  sudo journalctl -u researchgpt-backend -f"
echo "  tail -f backend/data/logs/backend.log"
echo ""
echo "Next steps:"
echo "  1. Create .env file (if not done): nano $APP_DIR/.env"
echo "  2. Set up GitHub Secrets for CI/CD (see below)"
echo ""
echo "GitHub Secrets needed:"
echo "  EC2_HOST      = $(curl -s http://169.254.169.254/latest/meta-data/public-ipv4 2>/dev/null || echo 'YOUR_EC2_IP')"
echo "  EC2_USERNAME  = $(whoami)"
echo "  EC2_SSH_KEY   = (paste your PEM private key)"
echo "============================================"
