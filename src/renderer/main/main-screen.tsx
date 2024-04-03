import { FC } from "react";
import { IconButton, Tabs } from "@radix-ui/themes";
import { HiOutlineCog6Tooth } from "react-icons/hi2";
import { Recorder } from "../recorder/recorder";
import { History } from "../history/history";
import { PageContainer } from "../common/page-container";
import { Postprocess } from "../postprocess/postprocess";
import { mainApi } from "../api";

export const MainScreen: FC = () => {
  return (
    <Tabs.Root defaultValue="record" style={{ height: "100%" }}>
      <PageContainer
        tabs={
          <Tabs.List style={{ flexGrow: "1" }}>
            <Tabs.Trigger value="record">Record</Tabs.Trigger>
            <Tabs.Trigger value="history">History</Tabs.Trigger>
            <Tabs.Trigger value="postprocess">Postprocessing</Tabs.Trigger>
          </Tabs.List>
        }
        statusButtons={
          <IconButton
            variant="surface"
            color="gray"
            onClick={() => mainApi.openSettingsWindow()}
          >
            <HiOutlineCog6Tooth />
          </IconButton>
        }
      >
        {/* TODO cant remember why i put asChild on the tabContent elements, but radix logs error if they exist, due to children not forwarding refs */}
        <Tabs.Content value="record">
          <Recorder />
        </Tabs.Content>

        <Tabs.Content value="history">
          <History />
        </Tabs.Content>

        <Tabs.Content value="postprocess">
          <Postprocess />
        </Tabs.Content>
      </PageContainer>
    </Tabs.Root>
  );
};
