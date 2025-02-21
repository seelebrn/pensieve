import { createStuffDocumentsChain } from "langchain/chains/combine_documents";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { OllamaEmbeddings } from "@langchain/community/embeddings/ollama";
import { ChatOpenAI, OpenAIEmbeddings } from "@langchain/openai";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { Document } from "@langchain/core/documents";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { createRetrievalChain } from "langchain/chains/retrieval";
import { ChatOllama } from "@langchain/community/chat_models/ollama";
import log from "electron-log/main";
import * as settings from "./settings";
import { RecordingTranscript, Settings } from "../../types";
import { isNotNull } from "../../utils";
import { getProgress, setProgress } from "./postprocess";
import { pullModel } from "./ollama";

const promptTemplate = `Be short and concise. 

Ci-joint la transcription d'une réunion. L'utilisateur sait que le contexte est une transcription de réunion et il n'y a pas besoin de le lui rappeler. "Moi" fait référence à l'utilisateur et "Ils" fait référence aux autres personnes présentes à la réunion.
L'information entre crochets décrit les timestamps auxquels le texte a été prononcé. 
Répond aux questions suivantes en te basant sur le contexte.:

<context>
{context}
</context>

Question: {input}`;

const prompts = {
  summary: "Génèse un résumé court du contenu de la réunion.",
  actionItems:
    "Extract action items from the meeting. Each action item should be a task that the user needs to follow up on after the meeting. " +
    "Provide one action item per line, in the format '{Me/They}: {action item text} ({timestamp})'.",
  sentenceSummary:
    "Résume le meeting très brièvement en une phrase de moins de 10 mots.",
};

const parseActionItems = (text: string) => {
  return text
    .split("\n")
    .map((line) => {
      const match = line.match(/^(.*?): (.*) \((\d*)\)$/);
      if (!match) {
        log.warn("Ignoring invalid action item line:", line);
        return null;
      }
      return {
        isMe: match[1].toLowerCase() === "me",
        action: match[2],
        time: parseInt(match[3], 10),
      };
    })
    .filter(isNotNull);
};

const getChatModel = async () => {
  const { llm } = await settings.getSettings();
  switch (llm.provider) {
    case "ollama":
      await pullModel(llm.providerConfig.ollama.chatModel.model);
      return new ChatOllama(llm.providerConfig.ollama.chatModel);
    case "openai": {
      return new ChatOpenAI(llm.providerConfig.openai.chatModel);
    }
    default:
      throw new Error(`Invalid LLM provider: ${llm.provider}`);
  }
};

const getEmbeddings = async () => {
  const { llm } = await settings.getSettings();

  switch (llm.provider) {
    case "ollama":
      await pullModel(llm.providerConfig.ollama.embeddings.model);
      return new OllamaEmbeddings(llm.providerConfig.ollama.embeddings);
    case "openai":
      return new OpenAIEmbeddings({
        ...llm.providerConfig.openai.embeddings,
        apiKey: llm.providerConfig.openai.chatModel.apiKey,
      });
    default:
      throw new Error(`Invalid LLM provider: ${llm.provider}`);
  }
};

const prepareContext = async (transcript: RecordingTranscript) => {
  const splitter = new RecursiveCharacterTextSplitter();
  const splitDocs = await splitter.splitDocuments([
    new Document({
      pageContent: transcript.transcription
        .map((t) => {
          const speakerText =
            t.speaker === "0"
              ? "They"
              : t.speaker === "1"
                ? "Me"
                : "Participant";
          return `${speakerText}: ${t.text} [${t.offsets.from}]`;
        })
        .join("\n"),
    }),
  ]);
  return splitDocs;
};

const prepareLangchain = async () => {
  const prompt = ChatPromptTemplate.fromTemplate(promptTemplate);
  const documentChain = await createStuffDocumentsChain({
    llm: await getChatModel(),
    prompt,
  });
  return documentChain;
};

const updateProgress = async (step: keyof Settings["llm"]["features"]) => {
  const { llm } = await settings.getSettings();
  if (!llm.features[step]) return;
  const total = Object.values(llm.features).filter((f) => f).length;
  setProgress("summary", (getProgress("summary") ?? 0) + 1 / total);
};

const summarizeWithEmbeddings = async (transcript: RecordingTranscript) => {
  const { llm } = await settings.getSettings();
  const context = await prepareContext(transcript);
  const chain = await prepareLangchain();

  const vectorstore = await MemoryVectorStore.fromDocuments(
    context,
    await getEmbeddings(),
  );
  const retriever = vectorstore.asRetriever();
  const retrievalChain = await createRetrievalChain({
    combineDocsChain: chain,
    retriever,
  });

  const summary = llm.features.summary
    ? await retrievalChain.invoke({
        input: prompts.summary,
      })
    : null;
  updateProgress("summary");
  const actionItems = llm.features.actionItems
    ? await retrievalChain.invoke({
        input: prompts.actionItems,
      })
    : null;
  updateProgress("actionItems");
  const sentenceSummary = llm.features.sentenceSummary
    ? await retrievalChain.invoke({
        input: prompts.sentenceSummary,
      })
    : null;
  updateProgress("sentenceSummary");
  return {
    summary: summary?.answer ?? null,
    actionItems: actionItems?.answer
      ? parseActionItems(actionItems.answer)
      : null,
    sentenceSummary: sentenceSummary?.answer ?? null,
  };
};

const summarizeWithContext = async (transcript: RecordingTranscript) => {
  const { llm } = await settings.getSettings();
  const context = await prepareContext(transcript);
  const chain = await prepareLangchain();

  const summary = llm.features.summary
    ? await chain.invoke({
        input: prompts.summary,
        context,
      })
    : null;
  updateProgress("summary");
  const actionItems = llm.features.actionItems
    ? await chain.invoke({
        input: prompts.actionItems,
        context,
      })
    : null;
  updateProgress("actionItems");
  const sentenceSummary = llm.features.sentenceSummary
    ? await chain.invoke({
        input: prompts.sentenceSummary,
        context,
      })
    : null;
  updateProgress("sentenceSummary");
  return {
    summary,
    actionItems: actionItems ? parseActionItems(actionItems) : null,
    sentenceSummary,
  };
};

export const summarize = async (transcript: RecordingTranscript) => {
  const { llm } = await settings.getSettings();
  return llm.useEmbedding
    ? summarizeWithEmbeddings(transcript)
    : summarizeWithContext(transcript);
};
