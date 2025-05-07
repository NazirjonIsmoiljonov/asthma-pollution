import streamlit as st

# ────────────────────────────────────────────────────────────
# ➊ MUST be the very first Streamlit call in your script:
st.set_page_config(
    page_title="Asthma Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)
# ────────────────────────────────────────────────────────────

import pandas as pd
import numpy as np
import plotly.express as px
import statsmodels.formula.api as smf

@st.cache_data
def load_data():
    # 1) AQI days
    aqi = pd.read_csv('annual_aqi_by_county_2021.csv')
    aqi['State'] = aqi['State'].str.strip()
    state_aqi = aqi.groupby('State').agg({
        'Days with AQI':'mean',
        'Days PM2.5':'mean',
        'Days Ozone':'mean',
        'Days NO2':'mean',
        'Days CO':'mean',
        'Days PM10':'mean'
    }).rename(columns={
        'Days with AQI':'AvgDaysWithAQI',
        'Days PM2.5':'AvgDaysPM25',
        'Days Ozone':'AvgDaysOzone',
        'Days NO2':'AvgDaysNO2',
        'Days CO':'AvgDaysCO',
        'Days PM10':'AvgDaysPM10'
    }).reset_index()

    # 2) Income‐weighted asthma prevalence & avg income
    inc = pd.read_csv('tableL6.csv')
    usps = {
      'AL':'Alabama','AK':'Alaska','AZ':'Arizona','AR':'Arkansas','CA':'California','CO':'Colorado',
      'CT':'Connecticut','DE':'Delaware','FL':'Florida','GA':'Georgia','HI':'Hawaii','ID':'Idaho',
      'IL':'Illinois','IN':'Indiana','IA':'Iowa','KS':'Kansas','KY':'Kentucky','LA':'Louisiana',
      'ME':'Maine','MD':'Maryland','MA':'Massachusetts','MI':'Michigan','MN':'Minnesota',
      'MS':'Mississippi','MO':'Missouri','MT':'Montana','NE':'Nebraska','NV':'Nevada',
      'NH':'New Hampshire','NJ':'New Jersey','NM':'New Mexico','NY':'New York',
      'NC':'North Carolina','ND':'North Dakota','OH':'Ohio','OK':'Oklahoma','OR':'Oregon',
      'PA':'Pennsylvania','RI':'Rhode Island','SC':'South Carolina','SD':'South Dakota',
      'TN':'Tennessee','TX':'Texas','UT':'Utah','VT':'Vermont','VA':'Virginia','WA':'Washington',
      'WV':'West Virginia','WI':'Wisconsin','WY':'Wyoming'
    }
    inc = inc[['State','Income','Weighted Numbere','Prevalence (Percent)']].copy()
    inc.columns = ['StateCode','IncomeBracket','WeightedN','PrevPct']
    inc = inc[inc['WeightedN'].str.replace(',','').str.match(r'^\d+$')]
    inc = inc[inc['IncomeBracket']!='Territories']
    inc['State']     = inc['StateCode'].map(usps)
    inc['WeightedN'] = inc['WeightedN'].str.replace(',','').astype(int)
    inc['PrevPct']   = inc['PrevPct'].astype(float)
    income_map = {
      '< $15,000':            7500,
      '$15,000–<$25,000':    20000,
      '$25,000–<$50,000':    37500,
      '$50,000–<$75,000':    62500,
      '>=$75,000':           87500
    }
    inc['IncomeMid'] = inc['IncomeBracket'].map(income_map)

    state_prev = inc.groupby('State') \
                    .apply(lambda g: np.average(g['PrevPct'], weights=g['WeightedN'])) \
                    .reset_index(name='AsthmaPrev')
    state_inc  = inc.groupby('State') \
                    .apply(lambda g: np.average(g['IncomeMid'], weights=g['WeightedN'])) \
                    .reset_index(name='AvgIncome')

    # 3) CO₂ per capita
    carbon = pd.read_excel('table4_shorter.xlsx', sheet_name='Table 4')
    carbon = carbon[['State','CarbonPerCapita2021']]
    carbon['State'] = carbon['State'].str.strip()

    # 4) Merge all into one DataFrame
    df = (state_prev
          .merge(state_inc, on='State')
          .merge(state_aqi,  on='State')
          .merge(carbon,     on='State'))
    return df

df = load_data()

# ────────────────────────────────────────────────────────────
# Title & Research Questions
st.title("📊 State‐Level Asthma & Pollution Dashboard (2021)")
st.markdown("""
**Research Questions**  
1. How does the number of days with poor air quality (AQI) relate to adult asthma prevalence?  
2. What role does average household income play in state asthma rates?  
3. Are CO₂ emissions or fine‐particle (PM₂.₅) exposure significant predictors of asthma?
""")

# ────────────────────────────────────────────────────────────
# Layout: two columns for AQI & Income charts
col1, col2 = st.columns(2)

with col1:
    fig1 = px.scatter(
        df, x='AvgDaysWithAQI', y='AsthmaPrev',
        size='AvgDaysWithAQI', color='State',
        title="Asthma Prevalence vs. Avg Days with AQI",
        labels={'AvgDaysWithAQI':'Average Days with AQI',
                'AsthmaPrev':'Asthma Prevalence (%)'}
    )
    st.plotly_chart(fig1, use_container_width=True)
    st.markdown("States with more days in the year having any AQI reading do **not** show a clear increase in adult asthma prevalence.")

with col2:
    fig2 = px.scatter(
        df, x='AvgIncome', y='AsthmaPrev',
        color='AvgIncome', trendline='ols',
        title="Asthma Prevalence vs. Avg Household Income",
        labels={'AvgIncome':'Avg Income (USD)','AsthmaPrev':'Asthma Prevalence (%)'},
        color_continuous_scale='Viridis'
    )
    st.plotly_chart(fig2, use_container_width=True)
    st.markdown("There's no obvious linear trend between higher income and lower asthma prevalence at the state level (p > 0.4).")

st.write("---")

# ────────────────────────────────────────────────────────────
# Full width: CO₂ bar chart & PM2.5 scatter
st.header("CO₂ Emissions & PM₂.₅ Exposure")

fig3 = px.bar(
    df.sort_values('CarbonPerCapita2021', ascending=False),
    x='State', y='CarbonPerCapita2021',
    title="Per‑Capita CO₂ Emissions by State (2021)",
    labels={'CarbonPerCapita2021':'CO₂ per Capita (metric tons)'}
)
fig3.update_layout(xaxis_tickangle=-45)
st.plotly_chart(fig3, use_container_width=True)
st.markdown("CO₂ emissions vary widely across states, but they alone do not explain differences in asthma prevalence (p > 0.2).")

fig4 = px.scatter(
    df, x='AvgDaysPM25', y='AsthmaPrev',
    trendline='ols', color='AvgDaysPM25',
    title="Asthma Prevalence vs. Days with PM₂.₅ Exposure",
    labels={'AvgDaysPM25':'Avg Days PM₂.₅','AsthmaPrev':'Asthma Prevalence (%)'},
    color_continuous_scale='Inferno'
)
st.plotly_chart(fig4, use_container_width=True)
st.markdown("More days with elevated PM₂.₅ show a slight negative slope, but it's not statistically significant in a linear model (p ≈ 0.6).")

st.write("---")

# ────────────────────────────────────────────────────────────
# Conclusion
st.header("🔍 Final Conclusion")
st.markdown("""
Across all of our state‑level linear models—AQI days, income, CO₂ emissions, and specific pollutant‑days—**none** of the covariates reached statistical significance (all p ≫ 0.05).  
🔹 **R² values** never rose above ~18 % (Adj R² near zero), indicating that state‑level aggregation washes out the environmental or socioeconomic signals driving asthma.  
**Next Steps:**  
- Move to county‑level or panel data across multiple years  
- Incorporate health/demographic covariates (smoking, age, healthcare access)  
- Use regularized or non‑linear models to handle multicollinearity  
""")

# ────────────────────────────────────────────────────────────
# Footer
st.markdown("""
---
**INFOSCI 301 | Duke Kunshan University, 2025**  
*Authors: Nazirjon Ismoiljonov & Jiean Zhou*
*Professor: Luyao Zhang*
""")
