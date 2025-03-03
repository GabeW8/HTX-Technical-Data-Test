import React from "react";
import {
  SearchProvider,
  SearchBox,
  Results,
  PagingInfo,
  Paging,
  Facet
} from "@elastic/react-search-ui";
import ElasticsearchAPIConnector from "@elastic/search-ui-elasticsearch-connector";
import "@elastic/react-search-ui-views/lib/styles/styles.css";

const connector = new ElasticsearchAPIConnector({
  host: "http://localhost:9200",
  index: "cv-transcriptions"
});

const config = {
  apiConnector: connector,
  searchQuery: {
    search_fields: {
      generated_text: {},
      duration: {},
      age: {},
      gender: {},
      accent: {}
    },
    result_fields: {
      generated_text: { raw: {} },
      duration: { raw: {} },
      age: { raw: {} },
      gender: { raw: {} },
      accent: { raw: {} }
    },
    facets: {
      gender: { type: "value" },
      age: {
        type: "range",
        ranges: [
          { from: 18, to: 30, name: "18-30" },
          { from: 31, to: 50, name: "31-50" },
          { from: 51, to: 100, name: "51+" }
        ]
      },
      accent: { type: "value" }
    }
  }
};

export default function App() {
  return (
    <SearchProvider config={config}>
      <div className="App">
        <h1>Transcription Search</h1>
        <SearchBox />
        <Facet field="gender" label="Gender" />
        <Facet field="age" label="Age" />
        <Facet field="accent" label="Accent" />
        <PagingInfo />
        <Results titleField="generated_text" />
        <Paging />
      </div>
    </SearchProvider>
  );
}
