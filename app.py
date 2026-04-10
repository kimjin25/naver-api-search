import streamlit as st # 스트림릿 라이브러리 임포트
import pandas as pd # 데이터 처리를 위한 판다스 임포트
import plotly.express as px # 시각화를 위한 plotly express 임포트
import plotly.graph_objects as go # 정밀 시각화를 위한 plotly graph_objects 임포트
import os # 시스템 환경변수 및 경로 관리를 위한 os 임포트
import json # JSON 데이터 파싱을 위한 json 임포트
import urllib.request # API 호출을 위한 urllib 임포트
import html # HTML 엔티티 변환을 위한 html 임포트
import re # 정규표현식 처리를 위한 re 임포트
from datetime import datetime, timedelta # 날짜 처리를 위한 클래스 임포트
from sklearn.feature_extraction.text import TfidfVectorizer # 텍스트 분석(TF-IDF)을 위한 라이브러리 임포트
from dotenv import load_dotenv # .env 파일 로드를 위한 라이브러리 임포트

# --- 환경 설정 및 초기화 ---
st.set_page_config(page_title="Naver Real-time Data Dashboard", layout="wide") # 대시보드 페이지 설정(제목, 넓게 보기)

# 시크릿 정보 로드 (Streamlit Cloud의 secrets 또는 로컬의 .env/os 환경변수 사용)
try:
    # Streamlit Cloud 배포 시 사용 (Setting > Secrets에 입력된 값 우선)
    CLIENT_ID = st.secrets["NAVER_CLIENT_ID"] 
    CLIENT_SECRET = st.secrets["NAVER_CLIENT_SECRET"]
except:
    # 로컬 환경에서 실행 시 .env 파일 로드
    load_dotenv() 
    CLIENT_ID = os.getenv("NAVER_CLIENT_ID") # 로컬 환경변수에서 ID 가져오기
    CLIENT_SECRET = os.getenv("NAVER_CLIENT_SECRET") # 로컬 환경변수에서 시크릿 가져오기

# --- 유틸리티 함수 정의 ---

def clean_text(text): # 텍스트 내 HTML 태그와 특수 기호를 제거하는 함수
    """HTML 태그 제거 및 엔티티 변환"""
    if not text: return "" # 텍스트가 없으면 빈 문자열 반환
    # HTML 태그 제거 및 특수 기호 정제 로직
    clean = re.sub(r'<[^>]+>', '', str(text)) # 정규표현식으로 모든 HTML 태그 제거
    clean = html.unescape(clean) # &quot; 같은 HTML 엔티티를 실제 문자로 변환
    return clean # 깨끗해진 텍스트 반환

@st.cache_data(ttl=3600) # 1시간 동안 API 호출 결과 캐싱하여 성능 최적화
def fetch_datalab_trend(keyword, start_date, end_date): # 네이버 데이터랩 트렌드 API 호출 함수
    """네이버 데이터랩 검색어 트렌드 실시간 수집"""
    url = "https://openapi.naver.com/v1/datalab/search" # API 엔드포인트 URL
    body = { # API 요청 바디 구성
        "startDate": start_date, "endDate": end_date, "timeUnit": "date",
        "keywordGroups": [{"groupName": keyword, "keywords": [keyword]}]
    }
    req = urllib.request.Request(url, data=json.dumps(body).encode("utf-8")) # 요청 객체 생성 및 인코딩
    req.add_header("X-Naver-Client-Id", CLIENT_ID) # 헤더에 ID 추가
    req.add_header("X-Naver-Client-Secret", CLIENT_SECRET) # 헤더에 시크릿 추가
    req.add_header("Content-Type", "application/json") # 콘텐츠 타입 설정
    try: # 오류 방지를 위한 예외 처리
        response = urllib.request.urlopen(req) # API 호출 실행
        res_data = json.loads(response.read().decode('utf-8')) # 결과 데이터를 JSON으로 로드
        data_list = [] # 데이터 저장용 리스트 최기화
        for result in res_data['results']: # 결과 리스트 순회
            for d in result['data']: # 일별 데이터 순회
                data_list.append({'date': d['period'], 'ratio': d['ratio'], 'keyword': keyword}) # 날짜와 비중 저장
        return pd.DataFrame(data_list) # 데이터프레임으로 변환하여 반환
    except Exception as e: # 오류 발생 시
        st.error(f"Trend API Error: {e}") # 에러 메시지 출력
        return pd.DataFrame() # 빈 데이터프레임 반환

@st.cache_data(ttl=1800) # 30분 동안 검색 결과 캐싱
def fetch_search_results(category, keyword, display=100): # 네이버 통합 검색 API 호출 함수
    """블로그, 뉴스, 카페, 쇼핑 검색 결과 실시간 수집"""
    mapping = {"blog": "blog", "news": "news", "cafe": "cafearticle", "shop": "shop"} # 카테고리 매핑 테이블
    api_node = mapping.get(category, "blog") # 매핑된 API 노드 가져오기
    encText = urllib.parse.quote(keyword) # 검색어 URL 인코딩
    url = f"https://openapi.naver.com/v1/search/{api_node}.json?query={encText}&display={display}" # API URL 생성
    req = urllib.request.Request(url) # 요청 객체 생성
    req.add_header("X-Naver-Client-Id", CLIENT_ID) # 헤더 설정
    req.add_header("X-Naver-Client-Secret", CLIENT_SECRET) # 헤더 설정
    try: # 예외 처리
        response = urllib.request.urlopen(req) # 호출 실행
        items = json.loads(response.read().decode('utf-8')).get('items', []) # 결과 아이템 리스트 추출
        df = pd.DataFrame(items) # 데이터프레임 생성
        if not df.empty: # 데이터가 있으면
            df['search_keyword'] = keyword # 검색 키워드 열 추가
            if 'lprice' in df.columns: # 쇼핑 데이터인 경우 가격 처리
                df['lprice'] = pd.to_numeric(df['lprice'], errors='coerce') # 가격을 숫자 형식으로 변환
        return df # 최종 결과 반환
    except Exception as e: # 에러 처리
        st.error(f"Search API ({category}) Error: {e}") # 메시지 출력
        return pd.DataFrame() # 빈 결과 반환

def extract_top_keywords(df, col='title', top_n=30): # 텍스트에서 주요 단어를 추출하는 함수
    if df.empty or col not in df.columns: return pd.DataFrame() # 데이터가 없으면 즉시 반환
    texts = df[col].fillna("").apply(clean_text) # 텍스트 정제 함수 적용
    if texts.str.strip().replace("", pd.NA).dropna().empty: return pd.DataFrame() # 유효한 텍스트가 없으면 반환
    vectorizer = TfidfVectorizer(max_features=1000, stop_words=['있는', '하는', '위한', '대한', '합니다', '입니다', '으로', '에서', '것입니다']) # TF-IDF 벡터라이저 설정
    try: # 분석 실행
        tfidf_matrix = vectorizer.fit_transform(texts) # 텍스트 학습 및 변환
        scores = tfidf_matrix.sum(axis=0).A1 # 단어별 가중치 합산
        words = vectorizer.get_feature_names_out() # 학습된 단어 리스트 가져오기
        return pd.DataFrame({'keyword': words, 'score': scores}).sort_values('score', ascending=False).head(top_n) # 상위 단어 선별 및 반환
    except: # 실패 시
        return pd.DataFrame() # 빈 결과 반환

# --- 대시보드 메인 UI ---
st.title("🚀 Naver Multi-Keyword Insights Dashboard") # 메인 제목 출력
st.markdown("여러 키워드를 비교 분석하고 실시간 데이터를 가져옵니다.") # 보조 설명 출력

# 사이드바 설정 영역
st.sidebar.header("⚙️ 설정") # 사이드바 헤더 추가
with st.sidebar.form("search_form"): # 입력 폼 생성 (불필요한 리프레시 방지)
    kw_input = st.text_input( # 키워드 입력창 생성
        "분석 키워드 입력", 
        value="", 
        placeholder="예: 봄, 야구", # 사용자 가이드용 힌트 텍스트
        help="여러 키워드 입력 시 콤마(,)로 구분해 주세요." # 도움말 툴팁 추가
    )
    target_kws = [k.strip() for k in kw_input.split(",") if k.strip()] # 입력된 텍스트를 리스트로 분리 및 정제
    
    default_start = datetime.now() - timedelta(days=365) # 시작 날짜 기본값 설정 (1년 전)
    default_end = datetime.now() # 종료 날짜 기본값 설정 (오늘)
    date_range = st.date_input("조회 기간 설정", value=(default_start, default_end)) # 날짜 범위 입력기 생성
    
    submit_button = st.form_submit_button("실시간 데이터 가져오기") # 제출 버튼 생성

if not CLIENT_ID or not CLIENT_SECRET: # API 키 유무 검사
    st.warning(".env 파일에 NAVER_CLIENT_ID와 SECRET을 설정해 주세요.") # 경고 메시지
    st.stop() # 실행 중단

# 검색 실행 로직
if submit_button and target_kws and len(date_range) == 2: # 버튼이 눌렸고 조건이 충족되면
    s_date = date_range[0].strftime('%Y-%m-%d') # 시작 날짜 형식 변환
    e_date = date_range[1].strftime('%Y-%m-%d') # 종료 날짜 형식 변환
    
    data_store = { # 수집된 데이터를 저장할 딕셔너리 구조 생성
        'trends': [], 'shops': {}, 'blogs': {}, 'news': {}, 'cafes': {},
        'kws': target_kws, 's_date': s_date, 'e_date': e_date
    }

    with st.spinner(f"키워드 데이터 수집 중..."): # 로딩 스피너 작동
        for kw in target_kws: # 입력된 모든 키워드에 대해 반복
            df_t = fetch_datalab_trend(kw, s_date, e_date) # 트렌드 수집
            if not df_t.empty: data_store['trends'].append(df_t) # 결과가 있으면 리스트에 추가
            data_store['blogs'][kw] = fetch_search_results("blog", kw) # 블로그 데이터 수집
            data_store['news'][kw] = fetch_search_results("news", kw) # 뉴스 데이터 수집
            data_store['cafes'][kw] = fetch_search_results("cafe", kw) # 카페 데이터 수집
            data_store['shops'][kw] = fetch_search_results("shop", kw) # 쇼핑 데이터 수집
        
        st.session_state['master_data'] = data_store # 수집 완료 후 세션 상태에 데이터 저장

# 분석 결과 렌더링 영역
if 'master_data' in st.session_state: # 세션에 데이터가 존재하면
    ds = st.session_state['master_data'] # 세션 데이터 매칭
    target_kws = ds['kws'] # 키워드 리스트 복구
    s_date, e_date = ds['s_date'], ds['e_date'] # 날짜 정보 복구
    
    tabs = st.tabs(["📊 통합 요약", "📈 트렌드 비교", "🛒 쇼핑/시장", "💬 소셜 인사이트", "🗓️ 원본 데이터"]) # 5개의 메인 탭 생성

    # 1번 탭: 요약 정보
    with tabs[0]:
        st.subheader("🔍 주요 지표 요약") # 소제목 출력
        for kw in target_kws: # 각 키워드별 카드 출력
            with st.expander(f"Keyword: {kw}", expanded=True): # 펼칠 수 있는 카드 섹션
                c1, c2, c3, c4 = st.columns(4) # 4개의 열로 데이터 배치
                trend_idx = -1 # 트렌드 데이터 인덱스 매칭용 변수
                for i, t in enumerate(ds['trends']): # 트렌드 데이터 리스트 탐색
                    if not t.empty and t['keyword'].iloc[0] == kw: # 키워드가 일치하면
                        trend_idx = i # 인덱스 저장
                        break
                
                c1.metric("트렌드 기록", len(ds['trends'][trend_idx]) if trend_idx != -1 else 0) # 수집된 트렌드 수 표시
                c2.metric("블로그", len(ds['blogs'][kw])) # 수집된 블로그 수 표시
                c3.metric("뉴스", len(ds['news'][kw])) # 수집된 뉴스 수 표시
                c4.metric("쇼핑 상품", len(ds['shops'][kw])) # 수집된 쇼핑 목록 수 표시
                
                if not ds['shops'][kw].empty: # 쇼핑 데이터가 있으면
                    avg_p = ds['shops'][kw]['lprice'].mean() # 평균 가격 계산
                    st.info(f"💡 {kw} 평균 가격: {avg_p:,.0f}원") # 정보를 알림 창으로 표시

    # 2번 탭: 시계열 트렌드 비교
    with tabs[1]:
        if ds['trends']: # 트렌드 데이터 리스트가 비어있지 않으면
            df_total_trend = pd.concat(ds['trends']) # 모든 키워드 데이터를 하나의 데이터프레임으로 병합
            df_total_trend['date'] = pd.to_datetime(df_total_trend['date']) # 날짜 형식 변환
            fig_trend = px.line(df_total_trend, x='date', y='ratio', color='keyword', title=f"키워드별 검색 트렌드 비교 ({s_date} ~ {e_date})") # 비교 선 그래프 생성
            st.plotly_chart(fig_trend, use_container_width=True) # 차트 출력
        else: st.warning("트렌드 데이터가 없습니다.") # 데이터 부족 시 경고

    # 3번 탭: 쇼핑 분석 및 추천
    with tabs[2]:
        sel_kw = st.selectbox("쇼핑 분석 키워드 선택", target_kws, key="sel_kw_shop") # 분석할 키워드 선택 박스
        df_s = ds['shops'][sel_kw] # 선택된 키워드의 쇼핑 데이터 가져오기
        if not df_s.empty: # 데이터가 존재하면
            col1, col2 = st.columns(2) # 화면을 반으로 나누기
            with col1: # 왼쪽: 가격 분포
                fig_price = px.histogram(df_s, x='lprice', title=f"[{sel_kw}] 가격 분포 히스토그램") # 히스토그램 생성
                st.plotly_chart(fig_price, use_container_width=True) # 출력
            with col2: # 오른쪽: 몰 비중
                mall_data = df_s['mallName'].value_counts().reset_index() # 판매몰 빈도 계산
                fig_mall = px.treemap(mall_data, path=['mallName'], values='count', title=f"[{sel_kw}] 판매몰 비중") # 트리맵 생성
                st.plotly_chart(fig_mall, use_container_width=True) # 출력
            
            st.markdown(f"#### 🎁 '{sel_kw}' 추천 상품 리스트") # 추천 섹션 소제목
            recom_df = df_s.sort_values('lprice').head(6) # 최저가 순 상위 6개 상품 추출
            cols = st.columns(3) # 3열 카드 레이아웃 생성
            for i, (_, row) in enumerate(recom_df.iterrows()): # 상품 순회
                with cols[i % 3]: # 열별 순차 배치
                    st.image(row['image'], use_container_width=True) # 상품 이미지 출력
                    st.markdown(f"**{clean_text(row['title'])}**") # 굵은 글씨로 상품명 표시
                    st.markdown(f"💰 {int(row['lprice']):,.0f}원 ({row['mallName']})") # 가격 및 판매처 출력
                    # 쇼핑 링크 정제 (HTML 엔티티 변환 및 https 보안 접속 강제)
                    target_url = html.unescape(row['link']).replace("http://", "https://") 
                    # 봇 탐지 회피를 위한 rel='noreferrer' 속성 추가 커스텀 버튼
                    st.markdown(f"""
                        <a href="{target_url}" target="_blank" rel="noreferrer" style="text-decoration: none;">
                            <div style="background-color: #ff4b4b; color: white; padding: 10px; border-radius: 5px; text-align: center; font-weight: bold;">
                                상품 보러가기
                            </div>
                        </a>
                    """, unsafe_allow_html=True)
                    st.markdown("---") # 구분선 추가
        else: st.warning(f"'{sel_kw}' 쇼핑 데이터가 없습니다.") # 데이터 부재 메시지

    # 4번 탭: 소셜 분석 및 핵심 콘텐츠
    with tabs[3]:
        sel_kw_social = st.selectbox("소셜 분석 키워드 선택", target_kws, key="sel_kw_social") # 키워드 선택
        ch_choice = st.radio("채널 선택", ["Blog", "News", "Cafe"], horizontal=True, key="ch_choice_social") # 채널 선택 라디오 버튼
        target_dict = {"Blog": ds['blogs'], "News": ds['news'], "Cafe": ds['cafes']}[ch_choice] # 채널에 해당하는 딕셔너리 찾기
        target_df = target_dict[sel_kw_social] # 선택된 데이터프레임 추출
        
        if not target_df.empty: # 데이터가 있으면
            col_left, col_right = st.columns([1, 1.2]) # 왼쪽: 통계, 오른쪽: 콘텐츠 리스트
            with col_left: # 왼쪽 영역
                st.subheader("📊 키워드 빈도 분석") # 헤더
                kw_res = extract_top_keywords(target_df) # TF-IDF 기반 단어 추출
                if not kw_res.empty: # 결과가 있으면
                    fig_kw = px.bar(kw_res, x='score', y='keyword', orientation='h', title=f"[{sel_kw_social}] 핵심 단어", color='score') # 가로 막대 그래프 생성
                    st.plotly_chart(fig_kw, use_container_width=True) # 출력
                else: st.write("분석 데이터가 부족합니다.") # 실패 시 안내
            
            with col_right: # 오른쪽 영역
                st.subheader(f"🔝 Top {ch_choice} 콘텐츠") # 콘텐츠 제목 출력
                for _, row in target_df.head(10).iterrows(): # 상위 10개 행 순회
                    title = clean_text(row['title']) # 제목 정제
                    st.markdown(f"📎 **[{title}]({row['link']})**") # 제목 및 하이퍼링크 출력
                    if 'description' in row: # 요약 설명이 있는 경우
                        desc = clean_text(row['description'])[:150] + "..." # 정제 및 글자수 제한
                        st.caption(desc) # 요약문 캡션 출력
                    st.markdown("---") # 구분선
        else: st.warning(f"'{sel_kw_social}' {ch_choice} 데이터가 없습니다.") # 부재 안내

    # 5번 탭: 원본 데이터 탐색
    with tabs[4]:
        sel_kw_raw = st.selectbox("데이터 조회 키워드 선택", target_kws, key="sel_kw_raw") # 키워드 선택
        df_type = st.selectbox("데이터 타입 선택", ["Blog", "News", "Cafe", "Shop"], key="df_type_raw") # 타입 선택
        raw_dict = {"Blog": ds['blogs'], "News": ds['news'], "Cafe": ds['cafes'], "Shop": ds['shops']} # 원본 데이터 사전
        st.dataframe(raw_dict[df_type][sel_kw_raw], use_container_width=True) # 원본 데이터프레임 출력
else: # 데이터가 아직 로드되지 않은 경우 초기 화면
    st.info("사이드바에서 키워드(예: 봄, 야구)를 입력하고 버튼을 눌러주세요.") # 안내 문구 출력
