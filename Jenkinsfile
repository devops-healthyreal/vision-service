pipeline {
    agent any

    environment {
        // ✅ 환경 변수 정의
        DOCKERHUB_CREDENTIALS = credentials('seung-dockerhub-credentials')  // 젠킨스에 등록된 DockerHub ID/PW
        DOCKER_IMAGE = "seung0208/vision-app"
        DEPLOY_USER = "ubuntu"
        DEPLOY_SERVER = "13.124.109.82"       // EC2 서버 IP (k3s 마스터)
        DEPLOY_PATH = "/home/ubuntu/k3s-deploy" // kubectl apply 실행 경로
        YAML_FILE = "k3s-app.yaml"             // 깃허브에 있는 yaml 파일 이름
    }

    stages {
        stage('Checkout') {
            steps {
                echo "📦 GitHub에서 소스코드 가져오기"
                checkout scm
            }
        }

        stage('Build Docker Image') {
            steps {
                echo "🐳 도커 이미지 빌드 중..."
                sh '''
                docker build -t ${DOCKER_IMAGE}:latest .
                '''
            }
        }

        stage('Login & Push Docker Image') {
            steps {
                echo "🚀 DockerHub 로그인 및 이미지 푸시"
                sh '''
                echo $DOCKERHUB_CREDENTIALS_PSW | docker login -u $DOCKERHUB_CREDENTIALS_USR --password-stdin
                docker push ${DOCKER_IMAGE}:latest
                '''
            }
        }

        stage('Sync YAML to Server') {
            steps {
                echo "🗂️ k3s-app.yaml 최신 버전을 서버로 동기화 (덮어쓰기 또는 신규 생성)"
                // 서버에 yaml 폴더가 없으면 만들고, yaml 파일 덮어쓰기
                sh """
                ssh -o StrictHostKeyChecking=no ${DEPLOY_USER}@${DEPLOY_SERVER} '
                    mkdir -p ${DEPLOY_PATH}
                '
                scp -o StrictHostKeyChecking=no ${YAML_FILE} ${DEPLOY_USER}@${DEPLOY_SERVER}:${DEPLOY_PATH}/${YAML_FILE}
                """
            }
        }

        stage('Deploy to k3s Cluster') {
            steps {
                echo "⚙️ 원격 서버에 배포(kubectl apply -f)"
                // SSH 플러그인 사용 or 직접 SSH 실행
                // kubectl set image <리소스종류>/<리소스이름> <deployment 내부에 정의한 컨테이너이름>=<새이미지> [옵션]
                sh """
                ssh -o StrictHostKeyChecking=no ${DEPLOY_USER}@${DEPLOY_SERVER} '
                    echo "🔄 최신 Docker 이미지 Pull..."
                    kubectl set image deployment/vision-app vision-container=${DOCKER_IMAGE}:latest --record || \
                    kubectl apply -f ${DEPLOY_PATH}/k3s-app.yaml
                    echo "✅ 배포 완료"
                '
                """
            }
        }
    }

    post {
        success {
            echo "🎉 배포 성공!"
        }
        failure {
            echo "❌ 배포 실패. 로그를 확인하세요."
        }
    }
}
