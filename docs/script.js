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
    
    drawEdges();
    
    nodes.forEach(node => {
        node.update();
        node.draw();
    });

    requestAnimationFrame(animate);
}

initNetwork();
animate();

// --- Interactive SVG Demo Logic ---
const btnSensory = document.getElementById('btn-sensory');
const btnTarget = document.getElementById('btn-target');
const btnReset = document.getElementById('btn-reset');

const pulseSensory = document.getElementById('pulse-sensory');
const pulseTarget = document.getElementById('pulse-target');

const path1 = document.getElementById('path-1');
const path2 = document.getElementById('path-2');

let sensoryActive = false;
let targetActive = false;

if(btnSensory) {
    btnSensory.addEventListener('click', () => {
        sensoryActive = true;
        pulseSensory.classList.add('anim-pulse-left');
        checkInterference();
    });

    btnTarget.addEventListener('click', () => {
        targetActive = true;
        pulseTarget.classList.add('anim-pulse-right');
        checkInterference();
    });

    function checkInterference() {
        if (sensoryActive && targetActive) {
            setTimeout(() => {
                // Constructive interference carves the canyon
                path1.classList.add('canyon');
                path2.classList.add('canyon');
                
                // Stop pulses after collision
                setTimeout(() => {
                    pulseSensory.classList.remove('anim-pulse-left');
                    pulseTarget.classList.remove('anim-pulse-right');
                }, 1000);
            }, 500); // Wait for them to meet in the middle
        }
    }

    btnReset.addEventListener('click', () => {
        sensoryActive = false;
        targetActive = false;
        pulseSensory.classList.remove('anim-pulse-left');
        pulseTarget.classList.remove('anim-pulse-right');
        path1.classList.remove('canyon');
        path2.classList.remove('canyon');
    });
}
