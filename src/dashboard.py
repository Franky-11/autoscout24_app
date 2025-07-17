import streamlit as st

from functions import*


st.title(":material/dashboard: Dashboard")
col1,col2=st.columns(2)
height=650

df = read_csv()
with col1:
    with st.container(border=True,height=height):
        col_cat, col_year = st.columns(2)
        with col_cat:
            with st.expander("Kategorie wählen"):
                cat = st.segmented_control("Kategorie", options=["make", "fuel", "gear", "model", "offerType"],
                                           default="make")
        with col_year:
            with st.expander("Erstzulassung wählen"):
                years = st.multiselect('Erstzulassung wählen:', options=list(range(2011, 2022)), default=[2011, 2021])

        tab_sales,tab_sales_volume=st.tabs(["Verkäufe", "Preis"])
        with tab_sales:
            sales = df[df["year"].isin(years)]["make"].size
           # st.metric(f"Verkäufe Gesamt", f"{sales:,} Tsd.")
           # st.markdown("**Anteil an Verkäufen**")
            plot_make_pie(df,cat,years)

        with tab_sales_volume:
            sales_volume = df[df["year"].isin(years)]["price"].sum()/1000000
           # st.metric(f"Preis Gesamt", f"{sales_volume:.2f} Mio.")
            plot_make_pie_sales_volume(df,cat,years)


with col2:
    with st.container(border=True,height=height):
        col_mark, col_year,col_gear,col_fuel = st.columns(4)
        with col_mark:
            st.markdown("**Top 10 Marken**")
        with col_year:
            years = st.multiselect("Erstzulassung", options=df["year"].unique().tolist()+["alle"], default=2011,key="year")
            years=[str(year) for year in years]
        with col_gear:
            gear = st.multiselect("Schaltung", options=df["gear"].unique().tolist(), default="Manual",key="gear")
        with col_fuel:
            fuel = st.multiselect("Kraftstoff", options=df["fuel"].unique().tolist(), default="Gasoline",key="fuel")

        top10=top_10_sales(df, fuel, gear, years)
        fig=plot_top_10(top10,years)
        st.plotly_chart(fig)


#------ Preis Zusammenhänge-----------#

with st.container(border=True,height=900):
    cols=st.columns(3)
    with cols[0]:
        fuel_price=st.multiselect("Kraftstoff",options=df["fuel"].unique().tolist(),default=["Gasoline","Diesel"])
    with cols[1]:
        make=st.multiselect("Marke",options=sorted(df["make"].unique().tolist()+["Alle"]),default=["Audi","Opel"])
    with cols[2]:
        gear_price=st.segmented_control("Schaltung",options=df["gear"].unique().tolist(),default=["Manual"],selection_mode="multi")

    tabs=st.tabs(["Mileage vs. Erstzulassung","Preis vs. Erstzulassung","Preis vs. Mileage"])
    with tabs[1]:
        st.markdown("**Preis vs. Erstzulassung**")
        _,df_filtered_iqr,_=df_outliers_removed(df,factor=1.5)
        df_filtered = df_filtered_iqr[(df_filtered_iqr["fuel"].isin(fuel_price)) & (df_filtered_iqr["gear"].isin(gear_price))]
        color_value = "make" if "Alle" not in make else None
        data_frame = df_filtered[df_filtered["make"].isin(make)] if "Alle" not in make else df_filtered
        fig=plot_rel_price_box(data_frame,color_value)
        st.plotly_chart(fig,use_container_width=True)

    with tabs[0]:
        st.markdown("**Mileage vs. Erstzulassung**")
        fig = plot_rel_price_box(data_frame, color_value, mileage=True)
        st.plotly_chart(fig, use_container_width=True)

    with tabs[2]:
        st.markdown("**Preis vs. Mileage**")
        fig = plot_rel_price_box(data_frame, color_value, price_mileage=True)
        st.plotly_chart(fig, use_container_width=True)








