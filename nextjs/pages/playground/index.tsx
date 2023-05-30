import { AppShell, Navbar } from "@mantine/core";
import React from "react";
import { useForm } from "react-hook-form";
import HeaderEvaluator, { MenuItem } from "../../components/HeaderEvaluator";
import Sidebar from "../../components/Sidebar";
import { FormValues } from "../../utils/types";
import Playground from "../../components/Playground";

const PlaygroundPage = () => {
  const form = useForm<FormValues>({
    defaultValues: {
      evalQuestionsCount: 5,
      chunkSize: 500,
      overlap: 100,
      splitMethod: "RecursiveTextSplitter",
      embeddingAlgorithm: "OpenAI",
      model: "wenxin",
      retriever: "similarity-search",
      gradingPrompt: "Descriptive",
      language: "zh-cn",
      numNeighbors: 3,
      files: [],
    },
  });

  return (
    <AppShell
      navbarOffsetBreakpoint="sm"
      navbar={
        <Navbar p="md" hiddenBreakpoint="sm" width={{ sm: 200, lg: 400 }}>
          <Sidebar form={form} />
          <br />
        </Navbar>
      }
      header={<HeaderEvaluator activeTab={MenuItem.Playground} />}
      styles={(theme) => ({
        main: {
          backgroundColor:
            theme.colorScheme === "dark"
              ? theme.colors.dark[8]
              : theme.colors.gray[0],
        },
      })}
    >
      <Playground form={form} />
    </AppShell>
  );
};
export default PlaygroundPage;
