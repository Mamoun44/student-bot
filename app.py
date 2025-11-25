import streamlit as st
# إصلاح مشكلة قاعدة البيانات في السيرفرات
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import os
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import CSVLoader

# عنوان الصفحة
st.set_page_config(page_title="مساعد الطلاب", page_icon="🎓")
st.title("🎓 اسألني عن الكلية")

# طلب المفتاح من المستخدم (أو يمكن وضعه في الإعدادات لاحقاً)
api_key = st.text_input("للتشغيل، أدخل مفتاح Groq API هنا:", type="password")

if api_key:
    # إعداد قاعدة البيانات والموديل
    @st.cache_resource
    def initialize_bot():
        # 1. تحويل النصوص لأرقام
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
        
        # 2. تحميل البيانات
        if os.path.exists("data.csv"):
            loader = CSVLoader(file_path='data.csv', encoding='utf-8')
            data = loader.load()
            # إنشاء قاعدة البيانات في الذاكرة المؤقتة
            vector_db = Chroma.from_documents(documents=data, embedding=embeddings)
            return vector_db
        return None

    # تشغيل البوت
    try:
        vector_db = initialize_bot()
        if vector_db:
            # إعداد العقل المدبر (Groq)
            llm = ChatGroq(groq_api_key=api_key, model_name="llama3-8b-8192")
            
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                retriever=vector_db.as_retriever(search_kwargs={"k": 3})
            )
            
            # واجهة الشات
            if "messages" not in st.session_state:
                st.session_state.messages = []

            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

            if prompt := st.chat_input("اكتب سؤالك هنا..."):
                st.session_state.messages.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.markdown(prompt)

                with st.chat_message("assistant"):
                    response = qa_chain.invoke({"query": prompt})
                    st.markdown(response['result'])
                    st.session_state.messages.append({"role": "assistant", "content": response['result']})
        else:
            st.error("لم يتم العثور على ملف data.csv")
            
    except Exception as e:
        st.error(f"حدث خطأ: {e}")