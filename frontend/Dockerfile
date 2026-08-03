# ── Stage 1: Build the React application ──────────────────────────────────────
FROM node:18-alpine AS builder

WORKDIR /app

# Copy package configurations
COPY package*.json ./
RUN npm install

# Copy source and configurations
COPY . .

# Build-time environment variable to map the FastAPI endpoint
ARG VITE_API_URL
ENV VITE_API_URL=$VITE_API_URL

# Build for production
RUN npm run build

# ── Stage 2: Serve using Nginx web server ────────────────────────────────────
FROM nginx:stable-alpine

# Copy compiled SPA dist files
COPY --from=builder /app/dist /usr/share/nginx/html

# Copy custom Nginx configuration to support SPA routing
COPY nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
