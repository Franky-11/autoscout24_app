import streamlit as st
from functions import read_csv_with_nan_duplicates,show_nan,df_outliers_removed,plot_outliers,get_outliers,image_path


df=read_csv_with_nan_duplicates()



st.image(image_path() ,width=500)
st.title(":material/dataset: Datensatz")

st.divider()
st.markdown(f"**Germany Cars Dataset from kaggle** &nbsp;&nbsp;|&nbsp;&nbsp;{df.shape[0]} Autos &nbsp;&nbsp;|&nbsp;&nbsp; {df.shape[1]} Features")
st.dataframe(df)


st.divider()
nan_total = df.isnull().sum().sum()
duplicates=df.duplicated().sum()
null_counts = df.isnull().sum().sort_values(ascending=False)
st.markdown(f"**NaN Werte**&nbsp;&nbsp;|&nbsp;&nbsp;{nan_total}&nbsp;&nbsp;|&nbsp;&nbsp;**Duplikate**&nbsp;&nbsp;|&nbsp;&nbsp;{duplicates}")
fig=show_nan(null_counts)
st.plotly_chart(fig,use_container_width=True)


st.divider()
st.markdown(
    "**Entfernen von Ausreißern (Preis,Mileage,HP) mittels IQR**")
cols=st.columns([1,0.5,2])
with cols[0]:
    st.markdown("""
    * Berechnen von Q1 und Q3 sowie IQR
    * Festlegen oberes Limit: Q3 + 1.5 × IQR  
    * Filtern der Daten nach oberem Limit
    """)
    st.write("")
    st.write("")
    factor = st.number_input("Faktor", 0.9, 2.1, 1.5, 0.1)
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)
    df_merged, df_filtered_iqr, iqr_bounds = df_outliers_removed(df, factor=factor)
    outliers = get_outliers(df_merged, df_filtered_iqr)
    st.markdown(f"{outliers["outliers"].sum()} Ausreißer")

with cols[2]:
    fig=plot_outliers(outliers,iqr_bounds)
    st.plotly_chart(fig,use_container_width=True)







