const canvas = document.getElementById('networkCanvas');
const ctx = canvas.getContext('2d');

let width, height;
let nodes = [];
let edges = [];

const NUM_NODES = 80;
const CONNECTION_DISTANCE = 150;

function resize() {
    width = window.innerWidth;
    height = window.innerHeight;
    canvas.width = width;
    canvas.height = height;
}

window.addEventListener('resize', resize);
resize();

class Node {
    constructor() {
        this.x = Math.random() * width;
        this.y = Math.random() * height;
        this.vx = (Math.random() - 0.5) * 0.5;
        this.vy = (Math.random() - 0.5) * 0.5;
        this.energy = 0;
        this.targetEnergy = 0;
    }

    update() {
        this.x += this.vx;
        this.y += this.vy;

        if (this.x < 0 || this.x > width) this.vx *= -1;
        if (this.y < 0 || this.y > height) this.vy *= -1;

        // Smooth energy transition
        this.energy += (this.targetEnergy - this.energy) * 0.1;
        if(this.targetEnergy > 0) this.targetEnergy -= 0.02; // dissipate
    }

    draw() {
        ctx.beginPath();
        ctx.arc(this.x, this.y, 2 + this.energy * 3, 0, Math.PI * 2);
        
        if (this.energy > 0.1) {
            ctx.fillStyle = `rgba(0, 240, 255, ${Math.min(1, this.energy)})`;
            ctx.shadowBlur = 15;
            ctx.shadowColor = '#00f0ff';
        } else {
            ctx.fillStyle = 'rgba(255, 255, 255, 0.2)';
            ctx.shadowBlur = 0;
        }
        
        ctx.fill();
    }
}

function initNetwork() {
    nodes = [];
    for (let i = 0; i < NUM_NODES; i++) {
        nodes.push(new Node());
    }
}

function drawEdges() {
    ctx.shadowBlur = 0;
    for (let i = 0; i < nodes.length; i++) {
        for (let j = i + 1; j < nodes.length; j++) {
            const dx = nodes[i].x - nodes[j].x;
            const dy = nodes[i].y - nodes[j].y;
            const dist = Math.sqrt(dx * dx + dy * dy);

            if (dist < CONNECTION_DISTANCE) {
                ctx.beginPath();
                ctx.moveTo(nodes[i].x, nodes[i].y);
                ctx.lineTo(nodes[j].x, nodes[j].y);
                
                // Edge energy is based on connected nodes
                const avgEnergy = (nodes[i].energy + nodes[j].energy) / 2;
                
                if (avgEnergy > 0.1) {
                    ctx.strokeStyle = `rgba(112, 0, 255, ${Math.min(0.8, avgEnergy + 0.2)})`;
                    ctx.lineWidth = 1 + avgEnergy * 2;
                } else {
                    ctx.strokeStyle = `rgba(255, 255, 255, ${0.1 - (dist / CONNECTION_DISTANCE) * 0.1})`;
                    ctx.lineWidth = 1;
                }
                
                ctx.stroke();
            }
        }
    }
}

// Simulate wave propagation randomly
setInterval(() => {
    const randomNode = nodes[Math.floor(Math.random() * nodes.length)];
    randomNode.targetEnergy = 2.0; // Trigger a wave
}, 1000);

function animate() {
    ctx.clearRect(0, 0, width, height);
    
    // Simulate Wave Propagation
    for (let i = 0; i < nodes.length; i++) {
        for (let j = i + 1; j < nodes.length; j++) {
            const dx = nodes[i].x - nodes[j].x;
            const dy = nodes[i].y - nodes[j].y;
            const dist = Math.sqrt(dx * dx + dy * dy);

            if (dist < CONNECTION_DISTANCE) {
                // Energy transfer between connected nodes (Simulated wave/diffusion)
                const diff = nodes[i].targetEnergy - nodes[j].targetEnergy;
                if (Math.abs(diff) > 0.05) {
                    const transfer = diff * 0.05; // Wave propagation speed
                    nodes[i].targetEnergy -= transfer;
                    nodes[j].targetEnergy += transfer;
                }
            }
        }
    }

    drawEdges();
    
    nodes.forEach(node => {
        node.update();
        node.draw();
    });

    requestAnimationFrame(animate);
}

// Funzione per iniettare energia
function injectEnergy(x, y, power, radius) {
    nodes.forEach(node => {
        const dx = node.x - x;
        const dy = node.y - y;
        const dist = Math.sqrt(dx*dx + dy*dy);
        if (dist < radius) {
            const blast = ((radius - dist) / radius) * power;
            node.targetEnergy += blast;
        }
    });
}

// Click / Touch esplosivo
window.addEventListener('pointerdown', (e) => {
    injectEnergy(e.clientX, e.clientY, 25.0, 250);
});

// Movimento del mouse lascia una scia
window.addEventListener('pointermove', (e) => {
    if (Math.random() > 0.5) { // throttled visual effect
        injectEnergy(e.clientX, e.clientY, 2.0, 100);
    }
});

initNetwork();
animate();
